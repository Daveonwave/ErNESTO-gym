import numpy as np
import pandas as pd
from tqdm import tqdm
from ernestogym.envs.single_agent.env import MicroGridEnv
from ernestogym.envs.single_agent.utils import parameter_generator
from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy
from stable_baselines3.common.callbacks import CheckpointCallback

pack_options = "ernestogym/ernesto/data/battery/pack.yaml"
ecm = "ernestogym/ernesto/data/battery/models/electrical/thevenin_fading_pack.yaml"
# ecm = "ernestogym/ernesto/data/battery/models/electrical/thevenin_pack.yaml"
r2c = "ernestogym/ernesto/data/battery/models/thermal/r2c_thermal_pack.yaml"
bolun = "ernestogym/ernesto/data/battery/models/aging/bolun_pack.yaml"
world = "ernestogym/envs/single_agent/world_fading.yaml"
#world = "ernestogym/envs/single_agent/world_deg.yaml"

# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(
    save_freq=100000,
    save_path="./logs/sac/models",
    name_prefix="sac",
    save_replay_buffer=True,
    save_vecnormalize=True,
)

test_profiles = ['70', '71', '72', '73', '74']
# test_profiles = ['70', '71', '72']
logdir = "./logs/fading/"
# logdir = "./logs/degradation/"


def run_sac():
    comparison_dict = {
        'actual_reward': {},
        'weighted_reward': {},
        'total_reward': {}
    }

    params = parameter_generator(battery_options=pack_options,
                                 electrical_model=ecm,
                                 thermal_model=r2c,
                                 aging_model=bolun,
                                 world_options=world
                                 )

    env = MicroGridEnv(settings=params)
    # env = Monitor(env, logdir, allow_early_resets=True)

    model = SAC(MlpPolicy, env, verbose=0, tensorboard_log="./logs/sac/tensorboard/")

    for _ in range(10):
        model.learn(total_timesteps=len(env.demand),
                    progress_bar=True,
                    # log_interval=5000,
                    # tb_log_name="fading_no_norm_clip_penalty",
                    callback=[checkpoint_callback],
                    reset_num_timesteps=False,
                    )
    print("################TRAINING is Done############")

    test_rewards = np.zeros(len(test_profiles))
    vec_env = model.get_env()

    for i in range(len(test_profiles)):
        vec_env.set_options({'eval_profile': test_profiles[i]})
        obs = vec_env.reset()

        for _ in tqdm(range(len(env.demand))):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = vec_env.step(action)
        test_rewards[i] = env.total_reward

        comparison_dict['total_reward'][test_profiles[i]] = env.total_reward
        comparison_dict['actual_reward'][test_profiles[i]] = env.actual_reward_list
        comparison_dict['weighted_reward'][test_profiles[i]] = env.weighted_reward_list

    df = pd.DataFrame.from_dict(comparison_dict)
    df.to_json(logdir + 'sac.json')


