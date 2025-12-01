#!/bin/bash

# Default seed (if not provided)
SEED=963

# Parse optional --seed argument
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --seed) SEED="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

#################################
# Define algorithms and models
#################################

# Algorithms to run
# ALGOS=("ppo" "random" "only_market" "battery_first")
ALGOS=("ppo")


# Models to test **for PPO only**
# PPO_MODELS=("20251120_456_partial_model")
# PPO_MODELS=("20251124_new_clipping_form_456")
# PPO_MODELS=("year_long_training_last_model")

# PPO_MODELS=("tanh_best_model_456")
PPO_MODELS=("huber_best_model_123")
# PPO_MODELS=("best_model_456_0005_alpha_huber")

#################################
# Run experiments
#################################

for ALGO in "${ALGOS[@]}"; do

    if [[ "$ALGO" == "ppo" ]]; then
        # Loop over all PPO models
        for MODEL in "${PPO_MODELS[@]}"; do
            echo "Running PPO with model: $MODEL"

            python launch_env.py \
                --algo "$ALGO" \
                --n_cores 1 \
                --exp_name "Test_per_report" \
                --replacement_cost 10.5 \
                --spread_factor 1 \
                --load_model "$MODEL" \
                --save_results_as "${ALGO}_${MODEL}" \
                --gamma 0.99 \
                --weight_trading 1 \
                --weight_degradation 1 \
                --weight_clipping 1 \
                --test \
                --world_settings ernestogym/envs/single_agent/ijcnn_deg_test_cell_scaling.yaml \
                --electrical_model ernestogym/ernesto/data/battery/models/electrical/thevenin_cell.yaml \
                --thermal_model ernestogym/ernesto/data/battery/models/thermal/r2c_thermal_cell.yaml \
                --aging_model ernestogym/ernesto/data/battery/models/aging/bolun_cell.yaml \
                --seed "$SEED"

            echo "Finished PPO with model $MODEL"
            echo "----------------------------------------"
        done

    else
        # Run non-PPO algorithms with default single model
        echo "Running algorithm: $ALGO"

        python launch_env.py \
            --algo "$ALGO" \
            --n_cores 1 \
            --exp_name "Test_per_report" \
            --replacement_cost 10.5 \
            --spread_factor 1 \
            --load_model "20251112_model" \
            --save_results_as "new_init_norm_${ALGO}" \
            --gamma 0.99 \
            --weight_trading 1 \
            --weight_degradation 1 \
            --weight_clipping 1 \
            --test \
            --world_settings ernestogym/envs/single_agent/ijcnn_deg_test_cell_scaling.yaml \
            --electrical_model ernestogym/ernesto/data/battery/models/electrical/thevenin_cell.yaml \
            --thermal_model ernestogym/ernesto/data/battery/models/thermal/r2c_thermal_cell.yaml \
            --aging_model ernestogym/ernesto/data/battery/models/aging/bolun_cell.yaml \
            --seed "$SEED"

        echo "Finished $ALGO"
        echo "----------------------------------------"
    fi
done
