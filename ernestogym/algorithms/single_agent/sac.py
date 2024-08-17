import os
import json
from tqdm import tqdm

from ernestogym.envs.single_agent.env import MicroGridEnv
from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy
from stable_baselines3.common.callbacks import CheckpointCallback


def train_sac(envs, args):
    print("######## SAC is running... ########")
    
    logdir = "./logs/" + args['exp_name']
    os.makedirs(logdir, exist_ok=True)
    
    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path="./logs/{}/models/".format(args['exp_name']),
        name_prefix="sac",
        save_replay_buffer=True,
        save_vecnormalize=True
        )
    
    model = SAC(MlpPolicy, 
                envs, 
                verbose=args['verbose'], 
                gamma=args['gamma'], 
                tensorboard_log="./logs/tensorboard/{}/sac/".format(args['exp_name']))

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
    
    
def eval_sac(env_params, args, test_profile, model_file="",):
    
    env = MicroGridEnv(settings=env_params)
    
    comparison_dict = {
        'test': test_profile,
        'pure_reward': {},
        'norm_reward': {},
        'weighted_reward': {},
        'total_reward': 0
    }
    
    logdir = "./logs/{}/results/sac/".format(args['exp_name'])
    os.makedirs(logdir, exist_ok=True)
    
    model_folder = "./logs/{}/models/".format(args['exp_name'])
    
    if not model_file:    
        #folder = "./logs/{}/models/".format(args['exp_name'])
        result_files = [f for f in os.listdir(model_folder) if os.path.isfile(os.path.join(folder, f)) and f.startswith("sac")]
        model_file = sorted(result_files)[-1]
        
    model = SAC.load(path=model_folder + model_file, env=env)
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
