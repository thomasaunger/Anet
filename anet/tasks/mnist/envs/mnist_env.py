import gym
import numpy as np
import torch
import torchvision

from gym import error, spaces, utils
from gym.utils import seeding
from torchvision import transforms

class MNISTEnv(gym.Env):
    def __init__(self, n=10, procs=0, proc_id=-1, train=True):
        assert  0 < procs,   "Must specify number of processes"
        assert -1 < proc_id, "Must specify process id"
        
        self.action_space = spaces.Discrete(10)
        
        self.observation_space = spaces.Box(
            low   = 0.0,
            high  = 1.0,
            shape = (28, 28, 1),
            dtype = "float32"
        )
        self.observation_space = spaces.Dict({
            "image": self.observation_space
        })
        
        self.n       = n
        self.procs   = procs
        self.proc_id = proc_id
        self.train   = train
        
        self.offset  = 0
    
    def step(self, action):
        if action % self.n == self.target:
            reward = 1.0
        else:
            reward = 0.0
        
        done = True
        
        return self.obs, reward, done, {}
    
    def reset(self):
        if not hasattr(self, "data"):
            data_loader = self.new_loader(self.np_random)
            self.data   = enumerate(data_loader)
        
        try:
            batch_idx, (data, target) = next(self.data)
        except StopIteration:
            data_loader = self.new_loader(self.np_random)
            self.data = enumerate(data_loader)
            batch_idx, (data, target) = next(self.data)
        
        image = np.zeros((28, 28, 1))
        image[:, :, 0] = data.squeeze().numpy()
        self.target = target
        
        #import matplotlib.pyplot as plt
        #plt.imshow(image[:, :, 0])
        #plt.title("target = " + str(self.target.item()))
        #plt.show()
        
        agent_dir = 0
        
        mission   = "unknown"
        
        # Observations are dictionaries containing:
        # - an image (partially observable view of the environment)
        # - the agent's direction/orientation (acting as a compass)
        # - a textual mission string (instructions for the agent)
        self.obs = {
            "image":     image,
            "direction": agent_dir,
            "mission":   mission
        }
        
        return self.obs
    
    def new_loader(self, np_random):
        transform = transforms.Compose([
            # you can add other transformations in this list
            transforms.ToTensor()
        ])
        
        mnist_data = torchvision.datasets.MNIST("", train=self.train, download=True, transform=transform)
        
        # Only select all digits below self.n
        idx = ~(mnist_data.targets == 0)
        for i in range(1, self.n):
            idx *= ~(mnist_data.targets == i)
        idx = ~idx
        mnist_data.data    = mnist_data.data[   idx]
        mnist_data.targets = mnist_data.targets[idx]
        
        # Random shuffle
        idx = torch.from_numpy(np_random.permutation(len(mnist_data.data)))
        mnist_data.data    = mnist_data.data[   idx][(self.proc_id + self.offset) % self.procs::self.procs]
        mnist_data.targets = mnist_data.targets[idx][(self.proc_id + self.offset) % self.procs::self.procs]
        
        # set offset for next loader
        self.offset = (self.offset + (self.procs - (len(idx) % self.procs))) % self.procs
        
        data_loader = torch.utils.data.DataLoader(mnist_data,
                                                  batch_size = 1)
        
        return data_loader

    def seed(self, seed=1337):
        # Seed the random number generator
        self.np_random, _ = seeding.np_random(seed)
        
        return [seed]
