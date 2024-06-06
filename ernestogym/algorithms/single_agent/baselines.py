from ernestogym.envs.single_agent.env import MicroGridEnv
from ernestogym.envs.single_agent.utils import parameter_generator
from tqdm import tqdm
import pandas as pd
import os

pack_options = "ernestogym/ernesto/data/battery/pack.yaml"
ecm = "ernestogym/ernesto/data/battery/models/electrical/thevenin_fading_pack.yaml"
# ecm = "ernestogym/ernesto/data/battery/models/electrical/thevenin_pack.yaml"
r2c = "ernestogym/ernesto/data/battery/models/thermal/r2c_thermal_pack.yaml"
bolun = "ernestogym/ernesto/data/battery/models/aging/bolun_pack.yaml"
world = "ernestogym/envs/single_agent/world_fading.yaml"
# world = "ernestogym/envs/single_agent/world_deg.yaml"

test_profiles = ['70', '71', '72', '73', '74']
# test_profiles = ['70', '71', '72']
logdir = "./logs/fading_norm_weights_trained_clip/"
# logdir = "./logs/degradation/"

os.makedirs(logdir, exist_ok=True)

weights = {"trading_coeff": 0.9, "operational_cost_coeff": 0.05, "degradation_coeff": 0, "clip_action_coeff": 0.05}

params = parameter_generator(battery_options=pack_options,
                             electrical_model=ecm,
                             thermal_model=r2c,
                             aging_model=bolun,
                             world_options=world,
                             use_reward_normalization=True,
                             reward_coeff=weights
                             )

env = MicroGridEnv(settings=params)


def random_action_policy(num_steps: int = None, filename='random'):
    comparison_dict = {
        'pure_reward': {},
        'actual_reward': {},
        'weighted_reward': {},
        'total_reward': {}
    }

    if num_steps is None:
        num_steps = len(env.demand)

    for profile in test_profiles:
        env.reset(options={'eval_profile': profile})

        for _ in tqdm(range(num_steps)):
            act = env.action_space.sample()  # Randomly select an action
            obs, reward, terminated, truncated, _ = env.step(act)  # Return observation and reward

            comparison_dict['pure_reward'][profile] = env.pure_reward_list
            comparison_dict['total_reward'][profile] = env.total_reward
            comparison_dict['actual_reward'][profile] = env.actual_reward_list
            comparison_dict['weighted_reward'][profile] = env.weighted_reward_list

    df = pd.DataFrame.from_dict(comparison_dict)
    df.to_json(logdir + filename + '.json')

    del comparison_dict
    del df


def deterministic_action_policy(action:float, filename: str, num_steps: int = None):
    assert 0 <= action <= 1, "The deterministic action must be between 0 and 1."

    comparison_dict = {
        'pure_reward': {},
        'actual_reward': {},
        'weighted_reward': {},
        'total_reward': {}
    }

    if num_steps is None:
        num_steps = len(env.demand)

    for profile in test_profiles:
        env.reset(options={'eval_profile': profile})

        for _ in tqdm(range(num_steps)):
            act = [action]  # Randomly select an action
            obs, reward, terminated, truncated, _ = env.step(act)  # Return observation and reward

            comparison_dict['pure_reward'][profile] = env.pure_reward_list
            comparison_dict['total_reward'][profile] = env.total_reward
            comparison_dict['actual_reward'][profile] = env.actual_reward_list
            comparison_dict['weighted_reward'][profile] = env.weighted_reward_list

    df = pd.DataFrame.from_dict(comparison_dict)
    df.to_json(logdir + filename + '.json')

    del comparison_dict
    del df
