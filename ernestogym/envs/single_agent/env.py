from typing import Any

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Dict, Box
from ernestogym.ernesto.energy_storage.bess import BatteryEnergyStorageSystem

from ernestogym.ernesto import EnergyGeneration, EnergyDemand, EnergyMarket


class MiniGridEnv(Env):
    """
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self,
                 settings: dict[str, Any],
                 ):
        """

        """
        # Build the battery object
        self._battery = BatteryEnergyStorageSystem(
            models_config=settings['models_config'],
            battery_options=settings['battery'],
            input_var=settings['input_var']
        )

        # Save the initialization bounds for environment parameters from which we will sample at reset time
        self._reset_params = settings['battery']['init']

        # Collect exogenous variables profiles
        self.demand = EnergyDemand(**settings["demand"])
        self.generation = EnergyGeneration(**settings["generation"]) if 'generation' in settings else None
        self.market = EnergyMarket(**settings["market"]) if 'market' in settings else None

        # Timing variables of the simulation
        self.timeframe = 0
        self.elapsed_time = 0
        self.iterations = 0

        # Fixed state variables which will be always present: 'demand', 'temperature' and 'soc'
        spaces = {
            'demand': Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            'temperature': Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'soc': Box(low=0, high=1, shape=(1,), dtype=np.float32),
        }

        # Add optional 'State of Health' in observation space
        if any(model['type'] == 'aging' for model in settings['models_config']):
            spaces['soh'] = Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # Add optional 'generation' in observation space
        if self.generation is not None:
            spaces['generation'] = Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)

        # Add optional 'bid' and 'ask' of energy market in observation space
        if self.market is not None:
            spaces['ask'] = Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
            spaces['bid'] = Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)

        if settings['rad_time']:
            spaces['day_time'] = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
            spaces['year_time'] = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        if settings['day_sec_time']:
            spaces['day'] = Box(low=0, high=365, shape=(1,), dtype=np.float32)
            spaces['seconds'] = Box(low=0, high=86400, shape=(1,), dtype=np.float32)

        self.observation_space = Dict(spaces=spaces)

        # Action Space: percentage of generated energy to store
        self.action_space = Box(low=0., high=1., dtype=np.float32, shape=(1,))

        self._total_reward = 0
        self._state = None
        self.state_list: list[np.ndarray] = []
        self.action_list: list[np.ndarray] = []

    def get_obs(self):
        return self._state

    def get_info(self):
        """
        Collect the information to pass to the environment after the execution of the step.
        """
        pass

    def reset(self, seed=None, options=None):
        """

        """
        super().reset(seed=seed, options=options)

        # Initialize the episode counter
        self.state_list = []
        self.action_list = []

        self.elapsed_time = 0
        self.iterations = 0

        # Initialize randomly the environment setting for a new run
        init_info = {key: np.random.uniform(low=value['low'], high=value['high']) for key, value in
                     self._reset_params.items()}

        demand_idx = np.random.randint(low=0, high=len(self.demand))
        self.timeframe = demand_idx * self.demand.timestep

        seconds_of_the_day = self.timeframe % (60*60*24)
        day_of_the_year = self.timeframe & (60*60*24*365)

        print(self.timeframe, seconds_of_the_day, day_of_the_year)
        exit()
        # generation_idx = self.generation.get_idx_from_time(time=self.timeframe)
        # market_idx = self.market.get_idx_from_time(time=self.timeframe)

        # TODO: create sin and cos of time

        self._battery.reset()
        self._battery.init(init_info=init_info)

    def step(self, action: np.ndarray):
        """

        """
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        assert self._state is not None, "Call reset before using step method."

        # normalize (action[0] + action[1]) on maximum of 1
        # get (1 - sum) which is the third implicit option
        # normalize action[1] and action[2] on (1-sum)


        reward = 0


        self._battery.step()

        self.k += 1
        terminated = bool()

    def render(self):
        pass

    def close(self):
        pass



if __name__ == "__main__":
    from ernestogym.envs.single_agent.utils import parameter_generator
    params = parameter_generator()
    env = MiniGridEnv(settings=params)
    state = env.get_obs()