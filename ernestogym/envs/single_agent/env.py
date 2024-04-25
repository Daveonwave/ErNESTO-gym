from typing import Any

from gymnasium import Env
from gymnasium.spaces import Dict, Box, Discrete
from ernestogym.ernesto.energy_storage.digital_twin.bess import BatteryEnergyStorageSystem

from ernestogym.ernesto import EnergyGeneration, EnergyDemand, EnergyMarket


class EnergyStorageEnv(Env):
    """
git
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self,
                 settings: dict[str, Any],
                 ):
        """

        """
        self._battery = BatteryEnergyStorageSystem(
            models_config=settings["models_config"],
            battery_options=settings["battery_options"],
            input_var=settings["input_var"]
        )

        self._generation_profile = EnergyGeneration(settings["generation"]) if 'generation' in settings else None
        self._demand_profile = EnergyDemand(settings["demand"]) if 'demand' in settings else None
        self._market = EnergyMarket(settings["market"]) if 'market' in settings else None

        self.observation_space = Dict({})
        self.action_space = Dict({})

    def _get_obs(self):
        pass

    def _get_info(self):
        pass

    def reset(self, seed=None, options=None):
        """

        """
        self._battery.reset()
        self._battery.init()

    def step(self, action):
        """

        """

        self._battery.step()

    def render(self):
        pass

    def close(self):
        pass
