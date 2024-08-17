import os
import json
from tqdm import tqdm

from ernestogym.envs.single_agent.env import MicroGridEnv
import ray
from ray.rllib.algorithms.sac.sac import SACConfig

def train_sac(envs, args):
    print("######## SAC is running... ########")
    config = SACConfig().training(
        gamma=args['gamma'], 
        lr=0.01, 
        train_batch_size=32
        )
    config = config.resources(num_gpus=0)
    config = config.env_runners(num_env_runners=len(envs))

    # Build a Algorithm object from the config and run 1 training iteration.
    algo = config.build(env=envs)
    save_result = algo.save()
    path_to_checkpoint = save_result.checkpoint.path
    
    algo.train()
    del algo
    
    
def eval_sac(env_params, args, test_profile, model_file="",):
    
    env = MicroGridEnv(settings=env_params)
    
    comparison_dict = {
        'test': test_profile,
        'pure_reward': {},
        'actual_reward': {},
        'weighted_reward': {},
        'total_reward': 0
    }
    
    logdir = "./logs/{}/results/sac/".format(args['exp_name'])
    os.makedirs(logdir, exist_ok=True)
    
    if not model_file:    
        folder = "./logs/{}/models/".format(args['exp_name'])
        result_files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f.startswith("sac")]
        model_file = sorted(result_files)[-1]
        
    model = SAC.load(path=folder + model_file, env=env)
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

    output_file = logdir + 'test_{}.json'.format(test_profile)

    with open(output_file, 'w', encoding ='utf8') as f: 
        json.dump(comparison_dict, f, allow_nan=False) 
