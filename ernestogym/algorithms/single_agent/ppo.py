import os
import json
from tqdm import tqdm
from typing import Callable

from ernestogym.envs.single_agent.env import MicroGridEnv
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import CheckpointCallback, StopTrainingOnMaxEpisodes, EvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func



def train_ppo(envs, args, eval_env_params, model_file=None):
    print("######## PPO is running... ########")
    
    logdir = "./logs/" + args['exp_name']
    os.makedirs(logdir, exist_ok=True)
    model_folder = "./logs/{}/models/".format(args['exp_name'])
    
    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=35000*5,
        save_path="./logs/{}/models/".format(args['exp_name']),
        name_prefix="ppo",
        save_replay_buffer=True,
        save_vecnormalize=True
        )
    
    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=args['n_episodes'], verbose=1)
    
    eval_env = MicroGridEnv(settings=eval_env_params)
    eval_callback = EvalCallback(eval_env, 
                                 best_model_save_path="./logs/{}/models/eval/".format(args['exp_name']),
                                 log_path="./logs/", 
                                 eval_freq=8760*5,
                                 n_eval_episodes=3,
                                 deterministic=True, 
                                 render=False)
    #stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=5, verbose=1)
    callbacks = [callback_max_episodes,]# eval_callback] # stop_train_callback]

    if model_file is None:
        model = PPO(policy=MlpPolicy, 
                    env=envs, 
                    verbose=args['verbose'], 
                    gamma=args['gamma'], 
                    tensorboard_log="./logs/tensorboard/ppo/{}".format(args['exp_name']),
                    ent_coef=0.01,
                    stats_window_size=1,
                    learning_rate=args['learning_rate']
                    )
    else:
        model = PPO.load(path=model_folder + model_file, env=envs)
        model.set_env(envs)
        print('Loaded model from: {}'.format(model_file))

    model.learn(total_timesteps=len(envs.get_attr("generation")[0]) * args['n_envs'] * args['n_episodes'],
                progress_bar=True,
                log_interval=args['log_rate'],
                tb_log_name="ppo_{}".format(args['exp_name']),
                callback=callbacks,
                reset_num_timesteps=True,
                )
        
    model.save("./logs/{}/models/{}".format(args['exp_name'], args['save_model_as']))
        
    print("######## TRAINING is Done ########")
    del model
    
    
def eval_ppo(env_params, args, test_profile, model_file=""):
    
    env = MicroGridEnv(settings=env_params)
        
    comparison_dict = {
        'test': test_profile,
        'pure_reward': {},
        'norm_reward': {},
        'weighted_reward': {},
        'total_reward': 0
    }
    
    logdir = "./logs/{}/results/{}/".format(args['exp_name'], args['save_results_as'])
    os.makedirs(logdir, exist_ok=True)
    
    model_folder = "./logs/{}/models/".format(args['exp_name'])

    if not model_file:   
        # Load the more recent model (last in alphabetical order) 
        result_files = [f for f in os.listdir(model_folder) if os.path.isfile(os.path.join(model_folder, f)) and f.startswith("ppo")]
        model_file = sorted(result_files)[-1]    
        
    model = PPO.load(path=model_folder + model_file, env=env)
    vec_env = model.get_env()
    
    vec_env.set_options({'eval_profile': test_profile})
    obs = vec_env.reset()
    
    done = False
    pbar = tqdm(total=len(vec_env.get_attr("generation")[0]))
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        done = dones[0]
        pbar.update(1)

    comparison_dict['total_reward'] = info[0]['total_reward']
    comparison_dict['pure_reward'] = info[0]['pure_reward_list']
    comparison_dict['norm_reward'] = info[0]['norm_reward_list']
    comparison_dict['weighted_reward'] = info[0]['weighted_reward_list']
    comparison_dict['actions'] = info[0]['actions']
    comparison_dict['states'] = info[0]['states']
    comparison_dict['traded_energy'] = info[0]['traded_energy']
    comparison_dict['soh'] = info[0]['soh']

    output_file = logdir + 'test_{}.json'.format(test_profile)

    with open(output_file, 'w', encoding ='utf8') as f: 
        json.dump(comparison_dict, f, allow_nan=False) 

