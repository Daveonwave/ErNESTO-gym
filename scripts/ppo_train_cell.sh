#!/bin/bash
python launch_env.py \
    --algo ppo \
    --n_envs 8 \
    --gamma 0.99 \
    --exp_name "20251104_ppo" \
    --spread_factor 1 \
    --save_model_as "20251104_ppo_RSE" \
    --n_episodes 100 \
    --weight_trading 1 \
    --weight_operational_cost 0 \
    --weight_degradation 1 \
    --weight_clipping 0.1 \
    --train \
    --log_rate 5 \
    --world_settings ernestogym/envs/single_agent/ijcnn_deg_train_cell.yaml \
    --electrical_model ernestogym/ernesto/data/battery/models/electrical/thevenin_cell.yaml \
    --thermal_model ernestogym/ernesto/data/battery/models/thermal/r2c_thermal_cell.yaml \
    --aging_model ernestogym/ernesto/data/battery/models/aging/bolun_cell.yaml
