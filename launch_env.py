import argparse

from ernestogym.algorithms.single_agent.ppo import run_ppo
from ernestogym.algorithms.single_agent.a2c import run_a2c
from ernestogym.algorithms.single_agent.sac import run_sac
from ernestogym.algorithms.single_agent.baselines import random_action_policy, deterministic_action_policy


def get_args():
    parser = argparse.ArgumentParser(description="ErNESTO-gym",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--battery_options")

    return vars(parser.parse_args())


if __name__ == '__main__':
    # run_ppo()
    # run_a2c()
    # run_sac()

    # random_action_policy()
    # deterministic_action_policy(action=0., filename="only_market")
    # deterministic_action_policy(action=1., filename="battery_first")
    # deterministic_action_policy(action=0.2, filename="20-80")
    # deterministic_action_policy(action=0.5, filename="50-50")

    run_ppo()
    run_a2c()
