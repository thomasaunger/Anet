from anet.tasks.mnist.envs.mnist_env import MNISTEnv

class MNISTEnvBinary(MNISTEnv):
    def __init__(self):
        MNISTEnv.__init__(self, 2)
