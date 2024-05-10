import gymnasium

from ernestogym.envs.single_agent.env import MiniGridEnv
from ernestogym.envs.single_agent.utils import parameter_generator

if __name__ == '__main__':
    params = parameter_generator()
    env = MiniGridEnv(settings=params)

    print(env.market[0])
    print(env.generation[0])
    print(env.demand[0])

    print(env.observation_space)

    env.reset(seed=42)
