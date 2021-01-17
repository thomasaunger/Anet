import gym
import numpy as np
import torch
import torchvision

from gym import error, spaces, utils
from gym.utils import seeding
from torchvision import transforms

transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor()
])

mnist_data = torchvision.datasets.MNIST("", train=True, download=True, transform=transform)

class MNISTEnv(gym.Env):
    
    def __init__(self):
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
        
        data_loader = torch.utils.data.DataLoader(mnist_data,
                                                  batch_size = 1,
                                                  shuffle    = True)
        self.data = enumerate(data_loader)
        #batch_idx, (example_data, example_targets) = next(self.data)
        
        #example_data_transformed = example_data.squeeze().numpy()
        
        #import matplotlib.pyplot as plt
        #plt.title(example_targets.item())
        #plt.imshow(example_data_transformed)
        #plt.show()
    
    def step(self, action):
        # Examples
        #reward    = 0
        #done      = False
        #image     = np.zeros((self.width, self.height, 3), dtype='uint8')
        #agent_dir = 0
        #mission   = ""
        
        if action == self.target:
            reward = 1.0
        else:
            reward = 0.0
        
        done = True
        
        return self.obs, reward, done, {}
    
    def reset(self):
        batch_idx, (data, target) = next(self.data)
        image = np.zeros((28, 28, 1))
        image[:, :, 0] = data.squeeze().numpy()
        self.target = target
        
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
