from gym.envs.registration import register

register(
    id="ArdaEnv-v0",
    entry_point="anet.tasks.hexagrid.envs:ArdaEnv",
)
