import gym
from multiprocessing import Process, Pipe

def get_local(obs):
    # get local view
    return obs["image"]*0.0

def reset(env):
    obs                   = env.reset()
    active_sender         = True
    active_receiver       = not active_sender
    active                = (active_sender, active_receiver)
    acting                = (False, active_receiver)
    sending               = (active_sender, False)
    globs                 = obs.copy()
    obs["image"]          = get_local(obs)
    obss                  = (globs, obs)
    extra                 = (0)
    return active, acting, sending, obss, extra

def step(env, action, prev_result):
    if prev_result[0][1]:
        # receiver's frame
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()
        active_sender           = True
        active_receiver         = not active_sender
        active                  = (active_sender, active_receiver)
        acting                  = (False, active_receiver)
        sending                 = (active_sender, False)
        globs                   = obs.copy()
        obs["image"]            = get_local(obs)
        obss                    = (globs, obs)
        extra = (0)
    else:
        # sender's frame
        reward          = 0.0
        done            = False
        active_sender   = False
        active_receiver = not active_sender
        active          = (active_sender, active_receiver)
        acting          = (False, active_receiver)
        sending         = (active_sender, False)
        obss            = prev_result[3]
        extra           = (0)
    return active, acting, sending, obss, extra, reward, done

def worker(conn, env):
    while True:
        cmd, action, prev_result = conn.recv()
        if cmd == "step":
            conn.send(step(env, action, prev_result))
        elif cmd == "reset":
            conn.send(reset(env))
        else:
            raise NotImplementedError

class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, env):
        assert len(env) >= 1, "No environment given."

        self.env               = env
        self.num_procs         = len(env)
        self.observation_space = self.env[0].observation_space
        self.action_space      = self.env[0].action_space
        
        self.locals = []
        self.processes = []
        for i, env in enumerate(self.env[1:]):
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, env))
            p.daemon = True
            p.start()
            remote.close()
            self.processes.append(p)

    def reset(self):
        for local in self.locals:
            local.send(("reset", None, None))
        self.prev_results = [reset(self.env[0])] + [local.recv() for local in self.locals]
        return zip(*self.prev_results)

    def step(self, actions):
        for local, action, prev_result in zip(self.locals, actions[1:, 1], self.prev_results[1:]):
            local.send(("step", action, prev_result))
        self.prev_results = [step(self.env[0], actions[0, 1], self.prev_results[0])] + [local.recv() for local in self.locals]
        return zip(*self.prev_results)

    def render(self):
        raise NotImplementedError

    def __del__(self):
        for p in self.processes:
            p.terminate()
