import os
# Suppress oneDNN and CPU info logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=INFO, 2=INFO+WARNING, 3=INFO+WARNING+ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # disables oneDNN optimizations messages
import argparse
from joblib import Parallel, delayed
from copy import deepcopy
from stable_baselines3.common.env_util import make_vec_env
from ernestogym.envs.single_agent.utils import parameter_generator
from ernestogym.algorithms.single_agent.ppo_new import train_ppo, eval_ppo
from ernestogym.algorithms.single_agent.a2c import train_a2c, eval_a2c
from ernestogym.algorithms.single_agent.sac import train_sac, eval_sac
from ernestogym.algorithms.single_agent.baselines import run_baseline
import cProfile, pstats, functools
from warnings import filterwarnings
from stable_baselines3.common.vec_env import SubprocVecEnv
import gymnasium as gym


filterwarnings(action='ignore')

algo_choices = ['ppo', 'a2c', 'sac', 'random', 'only_market', 'battery_first', '20-80', '50-50', '80-20', 'all_baselines']

def get_args():
    parser = argparse.ArgumentParser(description="ErNESTO-gym",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # RL algorithms and baselines
    parser.add_argument("--algo",  nargs=1, choices=algo_choices, help="")
    parser.add_argument("--exp_name", action="store", type=str, default='default')
    parser.add_argument("--n_envs", action="store", type=int, default=1)
    parser.add_argument("--n_episodes", action="store", type=int, default=1)
    parser.add_argument("--gamma", action="store", type=float, default=0.99)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--n_cores", action="store", type=int, default=1)
    parser.add_argument("--load_model", action="store", type=str, default=None)
    parser.add_argument("--save_model_as", action="store", type=str, default='')
    parser.add_argument("--save_results_as", action="store", type=str, default='')
    parser.add_argument("--spread_factor", action="store", type=float, default=1)
    parser.add_argument("--replacement_cost", action="store", type=float)
    
    # Environment configuration
    parser.add_argument("--battery_options", action="store", default="ernestogym/ernesto/data/battery/cell.yaml", help="")
    parser.add_argument("--electrical_model", action="store", default="ernestogym/ernesto/data/battery/models/electrical/thevenin_fading_pack.yaml",
                        type=str, help="")
    parser.add_argument("--thermal_model", action="store", default="ernestogym/ernesto/data/battery/models/thermal/r2c_thermal_cell.yaml",
                        type=str, help="")
    parser.add_argument("--aging_model", action="store", default="ernestogym/ernesto/data/battery/models/aging/bolun_pack.yaml",
                        type=str, help="")
    parser.add_argument("--world_settings", action="store", default="ernestogym/envs/single_agent/world_fading.yaml",
                        type=str, help="")
    parser.add_argument("--eval_world_settings", action="store", default="ernestogym/envs/single_agent/world_fading.yaml",
                        type=str, help="")
    
    parser.add_argument("--step", action='store', type=int)
    parser.add_argument("--seed", action='store', type=int)
    parser.add_argument("--random_battery_init", action='store_true')
    parser.add_argument("--random_data_init", action='store_true')
    
    # RL algorithms hyperparameters
    parser.add_argument("--learning_rate", action='store', type=float, default=0.00005)
    parser.add_argument("--policy_network", nargs="+", type=int, default=[64, 32])
    parser.add_argument("--log_std_init", action='store', type=float, default=-1)
    parser.add_argument("--batch_size", action='store', type=int, default=256)
    parser.add_argument("--n_steps", action='store', type=int, default=4096)
    parser.add_argument("--n_epochs", action='store', type=int, default=10)
    parser.add_argument("--clip_range", action='store', type=float, default=0.2)
    parser.add_argument("--gae_lambda", action='store', type=float, default=0.95)
    parser.add_argument("--ent_coef", action='store', type=float, default=0.0)
    parser.add_argument("--vf_coef", action='store', type=float, default=0.5)
    parser.add_argument("--max_grad_norm", action='store', type=float, default=0.5)
    
    # Evaluation arguments
    parser.add_argument("--eval_freq", action='store', type=int, default=8760)
    parser.add_argument("--n_eval_episodes", action='store', type=int, default=10)
    
    # Reward coefficients and normalization
    parser.add_argument("--weight_trading", action='store', type=float, default=1)
    parser.add_argument("--weight_degradation", action='store', type=float, default=1)
    parser.add_argument("--weight_clipping", action='store', type=float, default=0.1)
    parser.add_argument("--use_reward_normalization", action='store_true')
    
    # Utils
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--log_rate", action="store", type=int, default=500)
    
    return vars(parser.parse_args())


if __name__ == '__main__':
    
    # Profiler setup
    # profiler = cProfile.Profile()
    # profiler.enable()
    
    args = get_args()
    
    weights = {"trading_coeff": args['weight_trading'], 
               "degradation_coeff": args['weight_degradation'], 
               "clip_action_coeff": args['weight_clipping']
               }
    
    params = parameter_generator(world_options=args['world_settings'],
                                 battery_options=args['battery_options'],
                                 electrical_model=args['electrical_model'],
                                 thermal_model=args['thermal_model'],
                                 aging_model=args['aging_model'],
                                 use_reward_normalization=True,
                                 reward_coeff=weights,
                                 spread_factor=args['spread_factor'],
                                 replacement_cost=args['replacement_cost'] if 'replacement_cost' in args else None,
                                 seed=args['seed']
                                 )
    
    eval_params = parameter_generator(world_options=args['eval_world_settings'],
                                      min_soh=0.6,
                                      use_reward_normalization=False)
    
    base_seed = args.get("seed", 42)
    
    def make_env(rank):
        def _init():
            env_params = deepcopy(params)
            env_params["seed"] = int(base_seed) + int(rank)
            return gym.make("ernestogym/micro_grid-v1", settings=env_params)
        return _init
 
    if args['train']:  
        if args['algo'][0] == 'ppo':   
            envs = SubprocVecEnv([make_env(i) for i in range(args["n_envs"])])
            #envs = make_vec_env("ernestogym/micro_grid-v1", n_envs=args["n_envs"], env_kwargs={'settings': params}, vec_env_cls=SubprocVecEnv)
            train_ppo(envs, args, eval_env_params=eval_params, model_file=args['load_model'] if args['load_model'] else None)
            
        elif args["algo"][0] == 'a2c':
            envs = make_vec_env("ernestogym/micro_grid-v0", n_envs=args["n_envs"], env_kwargs={'settings':params})
            train_a2c(envs, args, model_file=args['load_model'] if args['load_model'] else None)
        
        elif args['algo'][0] == 'sac':
            envs = make_vec_env("ernestogym/micro_grid-v0", n_envs=args["n_envs"], env_kwargs={'settings':params})
            train_sac(envs, args, model_file=args['load_model'] if args['load_model'] else None)
    
        else:
            print("Algorithm not implemented!")
            exit(1)
        
        
    if args['test']: 
        if args['algo'][0] == 'ppo':   
            eval_func = eval_ppo
            
        elif args["algo"][0] == 'a2c':
            eval_func = eval_a2c
        
        elif args['algo'][0] == 'sac':
            eval_func = eval_sac
            
        else:
            eval_func = run_baseline

        test_profiles = [str(i) for i in range(370, 398)]
        n_cores = len(test_profiles) if args['n_cores'] >= len(test_profiles) else args['n_cores']
        Parallel(n_jobs=n_cores)(delayed(eval_func)(params, args, test, args['load_model']) for test in test_profiles)    
        
    # Profiler teardown
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumulative')
    # stats.dump_stats('microgrid.prof')