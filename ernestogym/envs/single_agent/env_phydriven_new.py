from typing import Any
from collections import OrderedDict, defaultdict
from copy import deepcopy

import numpy as np
from datetime import timedelta
from gymnasium import Env
from gymnasium.spaces import Box
from .rewards import operational_cost, linearized_degradation, soh_cost
from ernestogym.ernesto.energy_storage.bessPhyDriven import BatteryEnergyStorageSystemPhyDriven
from ernestogym.ernesto import PVGenerator, EnergyDemand, EnergyMarket, DummyGenerator, DummyMarket, AmbientTemperature, DummyAmbientTemperature

class MicroGridEnvPhyDriven(Env):
    """
    """
    SECONDS_PER_MINUTE = 60
    SECONDS_PER_HOUR = 60 * 60
    SECONDS_PER_DAY = 60 * 60 * 24
    DAYS_PER_YEAR = 365

    def __init__(self,
                 settings: dict[str, Any],
                 render_mode = None
                 ):
        """

        """
        metadata = {"render_modes": None}
        
        # Build the battery object
        self._battery = BatteryEnergyStorageSystemPhyDriven(
            models_config=settings['models_config'],
            battery_options=settings['battery'],
            input_var=settings['input_var'],
            check_soh_every=None
        )


        # Save the initialization bounds for environment parameters from which we will sample at reset time
        self._reset_params = settings['battery']['init']
        self._params_bounds = settings['battery']['bounds']
        # self._aging_options = settings['aging_options']
        self._random_battery_init = settings['random_battery_init']
        self._random_data_init = settings['random_data_init']
        self._seed = settings['seed']
        np.random.seed(self._seed)

        print(f"[INIT] Environment created with seed {self._seed}")

        # Collect exogenous variables profiles
        self.demand = EnergyDemand(**settings["demand"])
        self.generation = PVGenerator(**settings["generation"]) if 'generation' in settings \
            else DummyGenerator(gen_value=settings['dummy']['generation'])
        self.market = EnergyMarket(**settings["market"]) if 'market' in settings \
            else DummyMarket(**settings['dummy']["market"])
        self.temp_amb = AmbientTemperature(**settings["temp_amb"]) if 'temp_amb' in settings \
            else DummyAmbientTemperature(temp_value=settings['dummy']['temp_amb'])

        # Timing variables of the simulation
        self.timeframe = 0
        self.elapsed_time = 0
        self.iterations = 0
        '''Changed the _env_step in order to use dt_cycle and not dt'''
        self._env_step = settings['step_model']
        self.termination = settings['termination']
        self.termination['max_iterations'] = len(self.generation) - 1 if self.termination['max_iterations'] is None else self.termination['max_iterations']
        self.dt_previous_iter = self._env_step

        # Reward coefficients
        self._trading_coeff = settings['reward']['trading_coeff'] if 'trading_coeff' in settings['reward'] else 0
        self._op_cost_coeff = settings['reward']['operational_cost_coeff'] if 'operational_cost_coeff' in settings['reward'] else 0
        self._deg_coeff = settings['reward']['degradation_coeff'] if 'degradation_coeff' in settings['reward'] else 0
        self._clip_action_coeff = settings['reward']['clip_action_coeff'] if 'clip_action_coeff' in settings['reward'] else 0
        self._use_reward_normalization = settings['use_reward_normalization']
        self._trad_norm_term = None
        self._max_op_cost = None
        self.traded_energy = []
        
        # To distinguish between learning and testing
        self.eval_profile = None

        # MDP information
        self._state = None
        self.total_reward = 0
        self.state_list: list[np.ndarray] = []
        self.action_list: list[np.ndarray] = []
        # Reward without normalization and weights
        self.pure_rewards = {'r_trad':0, 'r_deg':0, 'r_clip': 0}
        # Normalized value of reward
        self.norm_rewards = {'r_trad':0, 'r_deg':0, 'r_clip':0}
        # Weighted value of reward multiplied by their coefficients
        self.weighted_rewards = {'r_trad':0, 'r_deg':0, 'r_clip':0}
        
        # Observation space support dictionary
        self.spaces = OrderedDict()
        self.spaces['temperature'] = {'low': 250., 'high': 400.}
        self.spaces['soc'] = {'low': 0., 'high': 1.}
        self.spaces['demand'] = {'low': 0., 'high': np.inf}
        self._obs_keys = ['temperature', 'soc', 'demand']

        # Add optional 'State of Health' in observation space
        if settings['soh']:
            # spaces['soh'] = Box(low=0, high=1, shape=(1,), dtype=np.float32)
            self._obs_keys.append('soh')
            self.spaces['soh'] = {'low': 0., 'high': 1.}

            # Add optional 'generation' in observation space
        if self.generation is not None:
            # spaces['generation'] = Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)s
            self._obs_keys.append('generation')
            self.spaces['generation'] = {'low': 0., 'high': np.inf}

        # Add optional 'bid' and 'ask' of energy market in observation space
        if self.market is not None:
            # spaces['ask'] = Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
            # spaces['bid'] = Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
            self._obs_keys.append('market')
            self.spaces['ask'] = {'low': 0., 'high': np.inf}
            self.spaces['bid'] = {'low': 0., 'high': np.inf}

        if settings['day_of_year']:
            # spaces['day_of_year'] = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
            self._obs_keys.append('day_of_year')
            self.spaces['sin_day_of_year'] = {'low': -1, 'high': 1}
            self.spaces['cos_day_of_year'] = {'low': -1, 'high': 1}

        if settings['seconds_of_day']:
            # spaces['seconds_of_day'] = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
            self._obs_keys.append('seconds_of_day')
            self.spaces['sin_seconds_of_day'] = {'low': -1, 'high': 1}
            self.spaces['cos_seconds_of_day'] = {'low': -1, 'high': 1}

        lows = [self.spaces[key]['low'] for key in self.spaces.keys()]
        highs = [self.spaces[key]['high'] for key in self.spaces.keys()]

        # Gym spaces
        self.observation_space = Box(low=np.array(lows), high=np.array(highs), dtype=np.float32)
        self.action_space = Box(low=0., high=1., dtype=np.float32, shape=(1,))

    def _get_obs(self) -> dict[str, Any]:
        """
        Collect the observation from the environment.
        Note that 'demand' and 'generation' are considered at the previous step or as a forecast of the actual values.
        """
        obs = {}

        for key in self._obs_keys:
            match key:
                case 'temperature':
                    obs['temperature'] = self._battery.get_temp()

                case 'soc':
                    obs['soc'] = self._battery.soc_series[-1]

                case 'demand':
                    idx = self.demand.get_idx_from_times(time=self.timeframe - self._env_step)
                    _, _, obs['demand'] = self.demand[idx]

                case 'soh':
                    obs['soh'] = self._battery.soh_series[-1]

                case 'generation':
                    idx = self.generation.get_idx_from_times(time=self.timeframe - self._env_step)
                    _, _, obs['generation'] = self.generation[idx]
                    print(idx)

                case 'market':
                    idx = self.market.get_idx_from_times(time=self.timeframe)
                    _, _, obs['ask'], obs['bid'] = self.market[idx]

                case 'day_of_year':
                    sin_year = np.sin(2 * np.pi / (self.SECONDS_PER_DAY * self.DAYS_PER_YEAR) * self.timeframe)
                    cos_year = np.cos(2 * np.pi / (self.SECONDS_PER_DAY * self.DAYS_PER_YEAR) * self.timeframe)
                    obs['sin_day_of_year'] = sin_year
                    obs['cos_day_of_year'] = cos_year

                case 'seconds_of_day':
                    sin_day = np.sin(2 * np.pi / self.SECONDS_PER_DAY * self.timeframe)
                    cos_day = np.cos(2 * np.pi / self.SECONDS_PER_DAY * self.timeframe)
                    obs['sin_seconds_of_day'] = sin_day
                    obs['cos_seconds_of_day'] = cos_day

                case 'energy_level':
                    obs['energy_level'] = self._battery.get_c_max() * self._battery.get_v() * self._battery.soc_series[-1]

                case _:
                    raise KeyError(f'Unknown observation variable: {key}')
        
        return obs

    def _get_actual_state(self) -> dict[str, Any]:
        """
        Collect the actual information regarding 'demand' and 'generation' to execute the step and compute the reward.

        This method retrieves the real-time values of demand and generation at the current timeframe to be used
        for environment dynamics and reward calculation.

        Returns:
            dict[str, Any]: A dictionary containing the actual 'demand' and 'generation' values.
        """
        actual_state = {}

        idx = self.demand.get_idx_from_times(time=self.timeframe)
        # idx_d = idx
        _, _, actual_state['demand'] = self.demand[idx]
        # actual_state['demand'] = actual_state['demand']*10
        if self.generation is not None:
            idx = self.generation.get_idx_from_times(time=self.timeframe)
            # idx_g = idx
            timestamp, _, actual_state['generation'] = self.generation[idx]
            # actual_state['generation'] = actual_state['generation']*10
        # print(idx_d,idx_g)
        return actual_state, timestamp
        
    def get_info(self) -> dict[str, Any]:
        """
        Collects and returns the main evaluation metrics and logged data.

        Returns:
            dict[str, Any]: All tracked variables during evaluation, including power, 
            demand, generation, market prices, reward history lists, and battery observations.
        """
        # info = {
        #     # Time series data collected during evaluation
        #     "power_list": getattr(self, "power_list", []),
        #     "demand_list": getattr(self, "demand_list", []),
        #     "generation_list": getattr(self, "generation_list", []),
        #     "price_ask_list": getattr(self, "price_ask_list", []),
        #     "price_bid_list": getattr(self, "price_bid_list", []),
        #     "pure_reward_list": getattr(self, "pure_reward_list", {}),
        #     "norm_reward_list": getattr(self, "norm_reward_list", {}),
        #     "weighted_reward_list": getattr(self, "weighted_reward_list", {}),
        # }
        info = {
            "pure_reward_list": getattr(self, "pure_reward_list", {}),
            "norm_reward_list": getattr(self, "norm_reward_list", {}),
            "weighted_reward_list": getattr(self, "weighted_reward_list", {}),
        }


        # Add battery observations if battery exists
        if hasattr(self, "_battery") and hasattr(self._battery, "get_observations"):
            info["battery_observations"] = self._battery.get_observations()

        return info



    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.

        This method resets the environment, including the battery system, reward collections, and timing variables.
        It also initializes the environment with random or predefined settings based on the configuration.

        Args:
            seed (int, optional): A seed for random number generation. Defaults to None.
            options (dict, optional): Additional options for resetting the environment. Defaults to None.

        Returns:
            tuple: A tuple containing the initial state and an empty info dictionary.
        """
        super().reset(seed=seed, options=options)
        
        self.total_reward = 0
        self._trad_norm_term = None
        self.elapsed_time = 0
        self.iterations = 0
        self.iseval = False
        self.pure_reward_list = defaultdict(list)
        self.norm_reward_list = defaultdict(list)
        self.weighted_reward_list = defaultdict(list)
        self.power_list = []
        self.cumulated_reward_list = []
        self.demand_list = []
        self.generation_list = []
        self.price_ask_list = []
        self.price_bid_list = []

        self.cumulated_reward = 0

        # Randomly sample a profile within the dataset
        if options is not None and 'eval_profile' in options:
            self.demand.profile = options['eval_profile']
            self.iseval = True
        else:
            self.demand.profile = np.random.choice(self.demand.labels)
        print("profile: ", self.demand.profile)

        # If seed is -1 we take datasets from the beginning
        if not self._random_data_init:
            gen_idx = 1
        # Otherwise we take an index between [1,len-1] so that we won't have out-of-index issues
        else:
            gen_idx = np.random.randint(low=1, high=len(self.generation) - self.termination['max_iterations'])
            # self._rng_gen_idx = np.random.default_rng(self._seed + int(self.demand.profile))

            '''Note to self: self.generation.__getitem__ require an index 
            and returns self_timestamps[idx], self._times[idx], self._history[idx]'''

            # gen_idx = self._rng_gen_idx.integers(low=1, high=len(self.generation) - self.termination['max_iterations'])
            # print(gen_idx)
        _, sampled_time, _ = self.generation[gen_idx]
        self.timeframe = sampled_time % (self.SECONDS_PER_DAY * self.DAYS_PER_YEAR)
        # print(gen_idx)
        # Initialize randomly the environment setting for a new run
        if self._random_battery_init:
            init_info = {key: np.random.uniform(low=value['low'], high=value['high']) for key, value in
                         self._params_bounds.items()}
            init_info['soh'] = 1
        else:
            init_info = {key: value for key, value in self._reset_params.items()}
            # init_info['voltage'] = self._battery.get_v()
            idx = self.temp_amb.get_idx_from_times(time=self.timeframe)
            # _, _, init_info['temperature'] = self.temp_amb[idx]
            # _, _, init_info['temp_ambient'] = self.temp_amb[idx]
            init_info['temp_ambient'] = 21.0+273.15
            init_info['temperature'] = init_info['temp_ambient']




        # Initialize the battery object
        self._battery.reset()
        self._battery.init(init_info=init_info)
                
        self._state = np.array(list(self._get_obs().values()), dtype=np.float32)        
        return self._state, {}

    def step(self, action: np.ndarray):
        """
        Perform a single step in the environment.

        This method updates the environment state based on the action taken by the agent. It computes the reward, 
        checks termination and truncation conditions, and returns the new state, reward, and additional information.

        Args:
            action (np.ndarray): The action taken by the agent, representing the fraction of energy to store.

        Returns:
            tuple: A tuple containing the new state, reward, termination flag, truncation flag, and info dictionary.
        """
        # Retrieve the actual amount of demand, generation and market
        obs = self._get_obs()
        actual_state, timestamp = self._get_actual_state()
        self.timeframe += self._env_step
        # print(action, obs)
        

        # Compute the fraction of energy to store/use and the fraction to sell/buy
        margin = actual_state['generation'] - actual_state['demand']
        print(margin)

        last_v = self._battery.get_v()
        i_max, i_min = self._battery.get_feasible_current(last_soc=self._battery.soc_series[-1], dt=self._env_step)

        # Clip the chosen action so that it won't exceed the SoC limits
        to_load = np.clip(a=margin * action[0], a_min=last_v * i_min, a_max=last_v * i_max)
        to_trade = margin - to_load
        
        # Current ambient temperature
        '''TO DO: WHEN T SENSOR ARE AVAILABLE'''
        # idx = self.temp_amb.get_idx_from_times(time=self.timeframe)
        # _, _, t_amb = self.temp_amb[idx]        
        t_amb = 21.0+273.15
                
        # Step of the battery model and update of internal state
        # for dtpiccolo:

        self._battery.step(load=to_load, dt=self._env_step, k=self.iterations, t_amb=t_amb,  dt_previous_iter = self.dt_previous_iter)
        #     get_i()

        self._battery.t_series.append(self.elapsed_time)
        self.elapsed_time += self._env_step
        print(self.elapsed_time)
        self.iterations += 1
                                
        # Termination condition
        terminated = bool(self._battery.soh_series[-1] <= self.termination['min_soh'])

        # Truncation conditions (due to the end of data)
        truncated = bool(
            (self.termination['max_iterations'] is not None and
             self.iterations >= self.termination['max_iterations'])
            or self.demand.is_run_out_of_data()
            or self.generation.is_run_out_of_data()
            or self.market.is_run_out_of_data()
        )

        # Trading reward with market and cost of degradation
        r_trading = to_trade * obs['ask'] * self._env_step/3600 if to_trade < 0 else to_trade * obs['bid'] * self._env_step/3600

        # Degradation penalty
        r_deg = -soh_cost(delta_soh=abs(self._battery.soh_series[-2] - self._battery.soh_series[-1]),
                          replacement_cost=self._battery.nominal_cost,
                          soh_limit=self.termination['min_soh'])
        
        # Clipping penalty from unfeasible actions
        # r_clipping = -abs(margin * action[0] - to_load)
        clip = margin * action[0] - to_load
        # r_clipping = -(0.1*clip**2)
        r_clipping = self.huber_penalty(c = clip)

        self.pure_rewards = {'r_trad': r_trading, 'r_deg': r_deg, 'r_clip': r_clipping}
        self._normalize_rewards(rewards=list(self.pure_rewards.values()))
        self.weighted_rewards = {'r_trad': self.norm_rewards['r_trad'] * self._trading_coeff,
                                 'r_deg': self.norm_rewards['r_deg'] * self._deg_coeff,
                                 'r_clip': self.norm_rewards['r_clip'] * self._clip_action_coeff}

        # Combining reward terms
        reward = sum(self.weighted_rewards.values())

        state = np.array(list(self._get_obs().values()), dtype=np.float32)
        info = {}

        if self.iseval:
            # self.cumulated_reward += reward
            # self.cumulated_reward_list.append(self.cumulated_reward)
            '''Commented'''
            self.power_list.append(to_load)
            self.demand_list.append(actual_state['demand'])
            self.generation_list.append(actual_state['generation'])
            self.price_ask_list.append(obs['ask'])
            self.price_bid_list.append(obs['bid'])
            ''''''
            for reward_type in ["pure", "norm", "weighted"]:
                reward_dict = getattr(self, f"{reward_type}_rewards")
                reward_list_dict = getattr(self, f"{reward_type}_reward_list")
                for k, v in reward_dict.items():
                    reward_list_dict[k].append(v)

            info = self.get_info()
            info['power_setpoint'] = to_load
            info['demand'] = obs ['demand']
            info['generation'] = obs ['generation']
            info['ask'] = obs ['ask']
            info['bid'] = obs ['bid']
            info['timestamp'] = timestamp
            # for k, v in obs.items():
            #     # if the key is new, initialize a list
            #     if k not in info:
            #         info[k] = []
            #         # save the current hour’s value
            #         info[k] = v
                # idx = self.demand.get_idx_from_times(time=self.timeframe)
                # print(self.demand.profile, idx)
        


        # if truncated or terminated:
        #     for key, values in self.norm_reward_list.items():
        #         plt.plot(values, label=key)

        #     plt.title("Rewards per Key")
        #     plt.xlabel("Index")
        #     plt.ylabel("Value")
        #     plt.legend()
        #     plt.grid(True)
        #     plt.show()
        
        return state, reward, terminated, truncated, info

    def _normalize_rewards(self, rewards: list):
        """
        Normalize reward values using min-max normalization.

        This method normalizes the reward components based on predefined terms and coefficients. 
        If normalization is disabled, the raw rewards are used as-is.

        Args:
            rewards (list): A list of raw reward values to be normalized.
        """
        if self._use_reward_normalization:
            if self._trad_norm_term is None:
                self._trad_norm_term = max(self.generation.max_gen * self.market.max_bid, 
                                           self.demand.max_demand * self.market.max_ask)
            
            self.norm_rewards['r_trad'] = rewards[0] / self._trad_norm_term
            self.norm_rewards['r_deg'] = rewards[1] 
            self.norm_rewards['r_clip'] = rewards[2] / max(abs(self.demand.max_demand - self.generation.min_gen), 
                                          abs(self.generation.max_gen - self.demand.min_demand))          
        else:
            self.norm_rewards['r_trad'] = rewards[0]
            self.norm_rewards['r_deg'] = rewards[1]
            self.norm_rewards['r_clip'] = rewards[2]

    def set_dt_previous_iter(self, dt):
        self.dt_previous_iter = dt

    def huber_penalty(self,c,k=0.1, alpha = 0.05):
        abs_c = abs(c)
        if abs_c <= k:
            return -alpha * 0.5 *abs_c * abs_c
        else:
            return -alpha * (k * (abs_c - 0.5*k))

