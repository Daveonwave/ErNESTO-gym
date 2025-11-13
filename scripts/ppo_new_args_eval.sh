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
    --n_cores 1 \
    --exp_name "20251112_PPO_full_experiment" \
    --replacement_cost 10.5 \
    --spread_factor 1 \
    --load_model "20251112_model" \
    --save_results_as "ppo_full" \
    --gamma 0.99 \
    --weight_trading 1 \
    --weight_degradation 1 \
    --weight_clipping 1 \
    --test \
    --world_settings ernestogym/envs/single_agent/ijcnn_deg_test_cell.yaml \
    --electrical_model ernestogym/ernesto/data/battery/models/electrical/thevenin_cell.yaml \
    --thermal_model ernestogym/ernesto/data/battery/models/thermal/r2c_thermal_cell.yaml \
    --aging_model ernestogym/ernesto/data/battery/models/aging/bolun_pack.yaml
