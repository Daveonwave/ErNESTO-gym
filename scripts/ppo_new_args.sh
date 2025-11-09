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
    --exp_name "Weekly_experiment" \
    --spread_factor 1 \
    --save_results_as "ppo_rse" \
    --gamma 0.99 \
    --n_episodes 100000 \
    --policy_network 64 32 \
    --log_std_init -1 \
    --batch_size 256 \
    --n_steps 4096 \
    --n_epochs 10 \
    --clip_range 0.2 \
    --gae_lambda 0.95 \
    --ent_coef 0.0 \
    --vf_coef 0.5 \
    --max_grad_norm 0.5 \
    --learning_rate 0.00005 \
    --log_rate 5 \
    --eval_freq 672 \
    --n_eval_episodes 1 \
    --seed "${SEED}" \
    --weight_degradation 0.1 \
    --world_settings ernestogym/envs/single_agent/ijcnn_deg_train_cell.yaml \
    --electrical_model ernestogym/ernesto/data/battery/models/electrical/thevenin_cell.yaml \
    --thermal_model ernestogym/ernesto/data/battery/models/thermal/r2c_thermal_cell.yaml\
    --aging_model ernestogym/ernesto/data/battery/models/aging/bolun_cell.yaml \
    --eval_world_settings ernestogym/envs/single_agent/ijcnn_deg_test_cell.yaml \
    --train
