import gym
from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX
from multiprocessing import Process, Pipe

def worker(conn, env):
    while True:
        cmd, action = conn.recv()
        if cmd == "step":
            active, acting, sending, obs, extra, reward, done = env.step(action)
            if done:
                active, acting, sending, obs, extra = env.reset()
            conn.send((active, acting, sending, obs, extra, reward, done))
        elif cmd == "reset":
            active, acting, sending, obs, extra = env.reset()
            conn.send((active, acting, sending, obs, extra))
        else:
            raise NotImplementedError

class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, envs):
        assert 1 <= len(envs), "No environment given."

        self.envs               = envs
        self.num_procs         = len(envs)
        self.observation_space = self.envs[0].observation_space
        self.action_space      = self.envs[0].action_space
        
        self.locals = []
        self.processes = []
        for i, env in enumerate(self.envs[1:]):
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, env))
            p.daemon = True
            p.start()
            remote.close()
            self.processes.append(p)

    def reset(self):
        for local in self.locals:
            local.send(("reset", None))
        results = [self.envs[0].reset()] + [local.recv() for local in self.locals]
        return zip(*results)

    def step(self, actions):
        for local, action in zip(self.locals, actions[1:]):
            local.send(("step", action))
        active, acting, sending, obs, extra, reward, done = self.envs[0].step(actions[0])
        if done:
            active, acting, sending, obs, extra = self.envs[0].reset()
        results = [(active, acting, sending, obs, extra, reward, done)] + [local.recv() for local in self.locals]
        return zip(*results)

    def render(self):
        raise NotImplementedError

    def __del__(self):
        for p in self.processes:
            p.terminate()
