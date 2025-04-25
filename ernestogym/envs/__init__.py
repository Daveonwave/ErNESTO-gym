from gymnasium.envs.registration import register

register(
    id='ernestogym/micro_grid-v0',
    entry_point='ernestogym.envs.single_agent.env:MicroGridEnv',
)

register(
    id='ernestogym/micro_grid_trading-v0',
    entry_point='ernestogym.envs.single_agent.env_trading:MicroGridEnv',
)
