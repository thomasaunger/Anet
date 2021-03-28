from gym.envs.registration import register

register(
    id="MNISTEnv-v0",
    entry_point="anet.tasks.mnist.envs:MNISTEnv",
)

register(
    id="MNISTEnvBinary-v0",
    entry_point="anet.tasks.mnist.envs:MNISTEnvBinary",
)

register(
    id="MNISTEnvQuaternary-v0",
    entry_point="anet.tasks.mnist.envs:MNISTEnvQuaternary",
)

register(
    id="MNISTEnvSenary-v0",
    entry_point="anet.tasks.mnist.envs:MNISTEnvSenary",
)

register(
    id="MNISTEnvOctonary-v0",
    entry_point="anet.tasks.mnist.envs:MNISTEnvOctonary",
)
