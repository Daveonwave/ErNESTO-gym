from gymnasium.envs.registration import register

register(
    id='ernestogym/MiniGrid-v0',
    entry_point='ernestogym.envs.single_agent.env:EnergyStorageEnv',
    nondeterministic=False
)
