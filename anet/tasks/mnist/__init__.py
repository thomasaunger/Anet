from gym.envs.registration import register

register(
    id="MNISTEnv-v0",
    entry_point="anet.tasks.mnist.envs:MNISTEnv",
)

register(
    id="MNISTEnvBinary-v0",
    entry_point="anet.tasks.mnist.envs:MNISTEnvBinary",
)
