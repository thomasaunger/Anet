import numpy as np
import torch

from anet.rl.algos import BaseAlgo

class PPOAlgo(BaseAlgo):
    def __init__(self, env, models, num_frames_per_proc=160, discount=0.99, lr=1e-4, beta1=0.9,
                 beta2=0.999, gae_lambda=0.99, entropy_coef=0.01, value_loss_coef=0.5,
                 max_grad_norm=0.5, recurrence=160, adam_eps=1e-5, clip_eps=0.2, epochs=4,
                 batch_size=5120, preprocess_obss=None, reshape_reward=None, use_comm=True, conventional=False, argmax=False, single_precision=False):
        
        super().__init__(env, models, num_frames_per_proc, discount, gae_lambda, preprocess_obss, reshape_reward, use_comm, conventional, argmax, single_precision)
        
        self.lr              = lr
        self.entropy_coef    = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm   = max_grad_norm
        self.recurrence      = recurrence
        self.clip_eps        = clip_eps
        self.epochs          = epochs
        self.batch_size      = batch_size
        
        if conventional:
            assert self.num_frames_per_proc % 2 == 0
        
        assert self.num_frames_per_proc % self.recurrence == 0
        
        assert self.batch_size          % self.recurrence == 0
        
        if batch_size <= num_frames_per_proc * env.num_procs:
            self.acc_batches = 1
            assert (num_frames_per_proc * env.num_procs) % batch_size == 0
        else:
            # Gradient accumulation.
            self.acc_batches = batch_size // (num_frames_per_proc * env.num_procs)
            assert batch_size % (num_frames_per_proc * env.num_procs) == 0
            assert epochs == 1
        
        self.optimizers = [torch.optim.Adam(model.parameters(), lr, (beta1, beta2), eps=adam_eps) for model in self.models]
        
        self.scaler = torch.cuda.amp.GradScaler()
    
    def update_parameters(self):
        # Initialize logs.
        logs = [{
            "return_per_episode":     [],
            "num_frames_per_episode": [],
            "num_frames":             0,
            "episodes_done":          0,
        } for m in range(self.num_agents)]
        
        # Initialize log values.
        log_entropies     = [[] for _ in range(self.num_agents)]
        log_values        = [[] for _ in range(self.num_agents)]
        log_policy_losses = [[] for _ in range(self.num_agents)]
        log_value_losses  = [[] for _ in range(self.num_agents)]
        log_grad_norms    = [[] for _ in range(self.num_agents)]
        log_losses        = [[] for _ in range(self.num_agents)]
        
        # Initialize batch values.
        batch_active      = torch.zeros(self.num_agents, device=self.device, dtype=torch.long)
        batch_entropy     = torch.zeros(self.num_agents, device=self.device)
        batch_value       = torch.zeros(self.num_agents, device=self.device)
        batch_policy_loss = torch.zeros(self.num_agents, device=self.device)
        batch_value_loss  = torch.zeros(self.num_agents, device=self.device)
        batch_loss        = torch.zeros(self.num_agents, device=self.device)
        
        for i in range(self.acc_batches):
            # Collect experiences.
            exps, acc_logs = self.collect_experiences()
            
            for m, log in enumerate(logs):
                log["return_per_episode"]     += acc_logs[m]["return_per_episode"]
                log["num_frames_per_episode"] += acc_logs[m]["num_frames_per_episode"]
                log["num_frames"]             += acc_logs[m]["num_frames"]
                log["episodes_done"]          += acc_logs[m]["episodes_done"]
            
            for _ in range(self.epochs):
                for inds in self._get_batches_starting_indexes():
                    memory = exps.memory[inds]
                    
                    # Initialize accumulative batch values.
                    acc_active      = torch.zeros(self.num_agents, device=self.device, dtype=torch.long)
                    acc_entropy     = torch.zeros(self.num_agents, device=self.device)
                    acc_value       = torch.zeros(self.num_agents, device=self.device)
                    acc_policy_loss = torch.zeros(self.num_agents, device=self.device)
                    acc_value_loss  = torch.zeros(self.num_agents, device=self.device)
                    acc_loss        = torch.zeros(self.num_agents, device=self.device)
                    
                    with torch.cuda.amp.autocast(enabled=not self.single_precision):
                        for j in range(self.recurrence):
                            # Create a sub-batch of experience.
                            sb = exps[inds + j]
                            
                            # Initialize values.
                            entropy  = torch.zeros(inds.shape[0], self.num_agents, device=self.device)
                            log_prob = torch.zeros(inds.shape[0], self.num_agents, device=self.device)
                            value    = torch.zeros(inds.shape[0], self.num_agents, device=self.device)
                            
                            # Compute loss.
                            for m, model in enumerate(self.models):
                                if torch.any(sb.active[:, m]):
                                    if self.use_comm:
                                        model_results = model(sb.obs[m][sb.active[:, m]], memory[sb.active[:, m], m]*sb.mask[sb.active[:, m], m], msg=sb.message[sb.active[:, m], 0])
                                    else:
                                        model_results = model(sb.obs[m][sb.active[:, m]], memory[sb.active[:, m], m]*sb.mask[sb.active[:, m], m])
                                    
                                    memory[sb.active[:, m], m] = model_results["memory"].float() ### casting necessary due to PyTorch Bug?
                                    dist                       = model_results["dist"]
                                    dists_speaker              = model_results["dists_speaker"]
                                    value[ sb.active[:, m], m] = model_results["value"].float() ### casting necessary due to PyTorch Bug?
                                    
                                    entropy[sb.active[:, m]*sb.acting[ :, m], m]  = dist.entropy()[sb.acting[sb.active[:, m], m]]
                                    entropy[sb.active[:, m]*sb.sending[:, m], m] += model.speaker_entropy(dists_speaker)[sb.sending[sb.active[:, m], m]]
                                    
                                    log_prob[sb.active[:, m]*sb.acting[ :, m], m]  = dist.log_prob(sb.action[sb.active[:, m], m])[sb.acting[sb.active[:, m], m]]
                                    log_prob[sb.active[:, m]*sb.sending[:, m], m] += model.speaker_log_prob(dists_speaker, sb.message[sb.active[:, m], m])[sb.sending[sb.active[:, m], m]]
                            
                            ratio       =  torch.exp(log_prob - sb.log_prob)
                            surr1       =  ratio * sb.advantage
                            surr2       =  torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                            policy_loss = -torch.min(surr1, surr2) * sb.active
                            
                            value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                            surr1         = (value - sb.returnn).pow(2)
                            surr2         = (value_clipped - sb.returnn).pow(2)
                            value_loss    = torch.max(surr1, surr2) * sb.active
                            
                            loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss
                            
                            # Update accumulative batch values.
                            acc_active      += sb.active.sum(  0)
                            acc_entropy     += entropy.sum(    0)
                            acc_value       += value.sum(      0)
                            acc_policy_loss += policy_loss.sum(0)
                            acc_value_loss  += value_loss.sum( 0)
                            acc_loss        += loss.sum(       0)
                            
                            # Update memories and messages for next epoch.
                            if 1 < self.epochs and j < self.recurrence - 1:
                                exps.memory[inds + j + 1] = memory.detach()
                    
                    # Update batch values.
                    batch_active      += acc_active
                    batch_entropy     += acc_entropy
                    batch_value       += acc_value
                    batch_policy_loss += acc_policy_loss
                    batch_value_loss  += acc_value_loss
                    batch_loss        += acc_loss
                    
                    # Update actor--critic.
                    self.scaler.scale(acc_loss.sum() / acc_active.sum()).backward()
                    if i == self.acc_batches - 1:
                        # Gradient accumulated.
                        [self.scaler.unscale_(optimizer) for optimizer in self.optimizers]
                        grad_norm = [sum(p.grad.data.norm(2) ** 2 for p in model.parameters() if p.grad is not None) ** 0.5 for model in self.models]
                        [torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm) for model in self.models]
                        [self.scaler.step(optimizer) for optimizer in self.optimizers]
                        self.scaler.update()
                        [optimizer.zero_grad() for optimizer in self.optimizers]
                        
                        # Update log values.
                        for m, log in enumerate(logs):
                            log_entropies[    m].append((batch_entropy[    m] / batch_active[m]).item() if 0 < batch_active[m] else 0)
                            log_values[       m].append((batch_value[      m] / batch_active[m]).item() if 0 < batch_active[m] else 0)
                            log_policy_losses[m].append((batch_policy_loss[m] / batch_active[m]).item() if 0 < batch_active[m] else 0)
                            log_value_losses[ m].append((batch_value_loss[ m] / batch_active[m]).item() if 0 < batch_active[m] else 0)
                            log_losses[       m].append((batch_loss[       m] / batch_active[m]).item() if 0 < batch_active[m] else 0)
                            log_grad_norms[   m].append((grad_norm[        m] / batch_active[m]).item() if 0 < batch_active[m] else 0)
                        
                        # Reinitialize batch values.
                        batch_active      = torch.zeros(self.num_agents, device=self.device, dtype=torch.long)
                        batch_entropy     = torch.zeros(self.num_agents, device=self.device)
                        batch_value       = torch.zeros(self.num_agents, device=self.device)
                        batch_policy_loss = torch.zeros(self.num_agents, device=self.device)
                        batch_value_loss  = torch.zeros(self.num_agents, device=self.device)
                        batch_loss        = torch.zeros(self.num_agents, device=self.device)
        
        # Log some values.
        for m, log in enumerate(logs):
            log["entropy"]     = np.mean(log_entropies[    m])
            log["value"]       = np.mean(log_values[       m])
            log["policy_loss"] = np.mean(log_policy_losses[m])
            log["value_loss"]  = np.mean(log_value_losses[ m])
            log["loss"]        = np.mean(log_losses[       m])
            log["grad_norm"]   = np.mean(log_grad_norms[   m])
        
        return logs
    
    def _get_batches_starting_indexes(self):
        indexes = np.arange(0, self.num_frames, self.recurrence)
        indexes = np.random.permutation(indexes)
        
        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [indexes[i:i + num_indexes] for i in range(0, len(indexes), num_indexes)]
        
        return batches_starting_indexes
