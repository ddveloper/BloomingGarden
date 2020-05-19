from gym.envs.registration import register

register(
    id='bloomingGarden-v0',
    entry_point='gym_bloomingGarden.envs:bloomingGardenEnv',
)
register(
    id='bloomingGarden-extrahard-v0',
    entry_point='gym_bloomingGarden.envs:bloomingGardenExtraHardEnv',
)