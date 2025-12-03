#!/bin/bash
# Default seed (if not provided)
SEED=42
# Parse optional --seed argument
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --seed) SEED="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done


python launch_env.py \
    --algo ppo \
    --n_envs 10 \
    --exp_name "weekly_expt_reward_rescale_lr2e4_longsteps" \
    --spread_factor 1 \
    --save_results_as "ppo_rscale_lr2e4_longsteps" \
    --gamma 0.995 \
    --n_episodes 100 \
    --policy_network 64 32 \
    --log_std_init -1 \
    --batch_size 128 \
    --n_steps 4096 \
    --n_epochs 7 \
    --clip_range 0.1 \
    --gae_lambda 0.98 \
    --ent_coef 0.001 \
    --vf_coef 1.0 \
    --max_grad_norm 0.5 \
    --learning_rate 0.0002 \
    --log_rate 5 \
    --eval_freq 35000 \
    --n_eval_episodes 10 \
    --seed "${SEED}" \
    --world_settings ernestogym/envs/single_agent/ijcnn_deg_train_cell_scaling.yaml \
    --electrical_model ernestogym/ernesto/data/battery/models/electrical/thevenin_cell.yaml \
    --thermal_model ernestogym/ernesto/data/battery/models/thermal/r2c_thermal_cell.yaml \
    --aging_model ernestogym/ernesto/data/battery/models/aging/bolun_cell.yaml \
    --eval_world_settings ernestogym/envs/single_agent/ijcnn_deg_test_cell_scaling.yaml \
    --train
