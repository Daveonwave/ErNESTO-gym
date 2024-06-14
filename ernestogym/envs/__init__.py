from gymnasium.envs.registration import register
from .single_agent.env import MicroGridEnv

register(
    id='ernestogym/micro_grid-v0',
    entry_point='ernestogym.envs.single_agent.env:MicroGridEnv',
)
