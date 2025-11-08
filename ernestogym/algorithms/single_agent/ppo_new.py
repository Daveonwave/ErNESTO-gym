import os
import json
import itertools
from tqdm import tqdm
from typing import Callable
import numpy as np

from ernestogym.envs.single_agent.env_new import MicroGridEnv
from ernestogym.envs.single_agent.env_phydriven import MicroGridEnvPhyDriven

from gymnasium import Wrapper
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import CheckpointCallback, StopTrainingOnMaxEpisodes, EvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
import math
import time


def cosine_schedule(initial_lr, total_timesteps):
    """
    Returns a cosine annealing learning rate schedule function.
    The LR will start at `initial_lr` and decay to 0 using cosine annealing.
    """
    def schedule(progress_remaining):
        # SB3 passes progress_remaining from 1.0 (start) to 0.0 (end)
        current_step = (1.0 - progress_remaining) * total_timesteps
        lr = initial_lr * 0.5 * (1 + math.cos(math.pi * current_step / total_timesteps))
        return lr

    return schedule


class ProfileInjectionEvalEnv(Wrapper):
    def __init__(self, env, demand_profiles, mode="cycle"):
        super().__init__(env)
        self.demand_profiles = demand_profiles
        self.mode = mode
        if mode == "cycle":
            self.profile_iterator = itertools.cycle(demand_profiles)

    def reset(self, **kwargs):
        # Select a profile
        if self.mode == "cycle":
            profile = next(self.profile_iterator)
        elif self.mode == "random":
            profile = np.random.choice(self.demand_profiles)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Inject it via the options dict
        options = kwargs.pop("options", {})
        options["eval_profile"] = profile
        return self.env.reset(options=options, **kwargs)


class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
    
    def _on_step(self) -> bool:
        info = self.locals["infos"][0]  # SB3 returns list of infos
        if "pure_rewards" in info:
            pr = info["pure_rewards"]
            # print(pr)
            self.logger.record("custom/reward_trading", pr.get('r_trad'))
            self.logger.record("custom/reward_degradation", pr.get('r_deg'))
            self.logger.record("custom/reward_clipping", pr.get('r_clip'))
        return True


def train_ppo(envs, args, eval_env_params, model_file=None):
    print("######## PPO is running... ########") 
    envs = VecNormalize(envs, norm_obs=True, norm_reward=False)
    
    logdir = "./logs/" + args['exp_name']
    os.makedirs(logdir, exist_ok=True)
    model_folder = "./logs/{}/models/seed_{}/".format(args['exp_name'],args['seed'])
    
    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=args['n_episodes'], verbose=1)
    callback_reward = RewardLoggerCallback()
    
    # Wrap the raw eval env in DummyVecEnv, then VecNormalize
    def make_eval_env():
        base_env = lambda: ProfileInjectionEvalEnv(
            env=Monitor(MicroGridEnv(settings=eval_env_params)),
            demand_profiles=[str(i) for i in range(370, 380)],
            mode="cycle"
        )
        return VecNormalize(DummyVecEnv([base_env]), training=False, norm_obs=True, norm_reward=True)
    
    eval_env = make_eval_env()
    eval_env.obs_rms = envs.obs_rms  # Sync normalization stats
    eval_env.ret_rms = envs.ret_rms  # (Optional) sync returns normalization
    eval_env.training = False        # Ensure no stats update during eval
    eval_env.norm_reward = False     # Often preferred during evaluation
    
    eval_callback = EvalCallback(eval_env, 
                                 best_model_save_path="./logs/{}/models/eval/seed_{}/".format(args['exp_name'],args['seed']),
                                 log_path="./logs/{}/seed_{}/".format(args['exp_name'],args['seed']), 
                                 eval_freq=args['eval_freq'],
                                 n_eval_episodes=args['n_eval_episodes'],
                                 deterministic=True, 
                                 render=False)
    
    callbacks = [callback_max_episodes, eval_callback, callback_reward]
    
    if model_file is not None:
        model = PPO.load(path=model_folder + model_file, env=envs)
        model.set_env(envs)
        print('Loaded model from: {}'.format(model_file))
    else:
        model = PPO("MlpPolicy", 
                    env=envs, 
                    gamma=args['gamma'], 
                    policy_kwargs=dict(net_arch=args['policy_network'], log_std_init=args['log_std_init']),
                    batch_size=args['batch_size'],
                    n_steps=args['n_steps'],
                    n_epochs=args['n_epochs'],
                    gae_lambda=args['gae_lambda'],
                    clip_range=args['clip_range'],
                    ent_coef=args['ent_coef'],
                    vf_coef=args['vf_coef'],
                    max_grad_norm=args['max_grad_norm'],
                    tensorboard_log="./logs/tensorboard/{}/ppo/".format(args['exp_name']),
                    #stats_window_size=1,
                    learning_rate=cosine_schedule(args['learning_rate'], envs.get_attr("termination")[0]['max_iterations'] * args['n_envs'] * args['n_episodes']),
                    verbose=args['verbose']
                    )
        model.set_env(envs)

    model.learn(total_timesteps=envs.get_attr("termination")[0]['max_iterations'] * args['n_envs'] * args['n_episodes'],
                progress_bar=True,
                log_interval=args['log_rate'],
                tb_log_name="seed_{}".format(args['seed']),
                callback=callbacks,
                reset_num_timesteps=True,
                )
        
    model.save("./logs/{}/{}/models/{}".format(args['exp_name'], args['seed'], args['save_model_as']))
    print("######## TRAINING is Done ########")
    
    
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

def eval_ppo_phydriven(env_params, args, test_profile, model_file=""):
    
    env = MicroGridEnvPhyDriven(settings=env_params)
        
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


    dt_cycle = env_params['step_model']
    dt_RL = env_params['step']
    # # Compute number of cycles (ensure integer multiple)
    # ratio = dt_RL / dt_cycle
    # assert abs(ratio - round(ratio)) < 1e-9, "dt must be a multiple of dt_cycle"
    # iter_cycles = int(round(ratio))

    while not done:
        ''' Scelta dell'azione'''
        action, _states = model.predict(obs)

        '''Applicazione dell'azione per un tempo dt_RL'''
        end_time = time.time() + 0.5
        while time.time() < end_time:
            # start_it_time = time.time()
            obs, rewards, dones, info = vec_env.step(action)
            if dones[0]:
                done = True
                break

        # vec_env.timeframe += dt_RL
        pbar.update(1)
    env._battery._electrical_model._cycler.stop_follow_P()
    env._battery._electrical_model._cycler.exit_communications()

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


