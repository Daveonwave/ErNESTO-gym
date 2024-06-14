import os
import json
from tqdm import tqdm

from ernestogym.envs.single_agent.env import MicroGridEnv
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat


class SummaryWriterCallback(BaseCallback):
    '''
    Snippet skeleton from Stable baselines3 documentation here:
    https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html#directly-accessing-the-summary-writer
    '''

    def _on_training_start(self):
        self._log_freq = 10  # log every 10 calls

        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

    def _on_step(self) -> bool:
        '''
        Log my_custom_reward every _log_freq(th) to tensorboard for each environment
        '''
        if self.n_calls % self._log_freq == 0:
            rewards = self.locals['my_custom_info_dict']['my_custom_reward']
            for i in range(self.locals['env'].num_envs):
                self.tb_formatter.writer.add_scalar("rewards/env #{}".format(i+1),
                                                     rewards[i],
                                                     self.n_calls)


def train_ppo(envs, args):
    print("######## PPO is running... ########")
    
    logdir = "./logs/" + args['exp_name']
    os.makedirs(logdir, exist_ok=True)
    
    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=525000,
        save_path="./logs/{}/models/".format(args['exp_name']),
        name_prefix="ppo",
        save_replay_buffer=True,
        save_vecnormalize=True
        )
    
    model = PPO(MlpPolicy, 
                envs, 
                verbose=args['verbose'], 
                gamma=args['gamma'], 
                tensorboard_log="./logs/tensorboard/{}/ppo/".format(args['exp_name']))

    for i in range(args['n_episodes']):
        model.learn(total_timesteps=len(envs.get_attr("demand")[0]) * args['n_envs'],
                    progress_bar=True,
                    log_interval=args['log_rate'],
                    tb_log_name="ep_{}".format(i),
                    callback=[checkpoint_callback],
                    reset_num_timesteps=False,
                    )
        
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
    
    logdir = "./logs/{}/results/ppo/".format(args['exp_name'])
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

    for _ in tqdm(range(len(env.demand))):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
    
    comparison_dict['total_reward'] = env.total_reward
    comparison_dict['pure_reward'] = env.pure_reward_list
    comparison_dict['norm_reward'] = env.norm_reward_list
    comparison_dict['weighted_reward'] = env.weighted_reward_list
    comparison_dict['actions'] = [action.tolist() for action in env.action_list]
    comparison_dict['states'] = [state.tolist() for state in env.state_list]

    output_file = logdir + 'test_{}.json'.format(test_profile)

    with open(output_file, 'w', encoding ='utf8') as f: 
        json.dump(comparison_dict, f, allow_nan=False) 

