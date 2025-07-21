python launch_env.py \
    --algo battery_first \
    --n_cores 5 \
    --exp_name "ijcnn_presentation" \
    --spread_factor 1 \
    --weight_trading 1 \
    --weight_operational_cost 0 \
    --weight_degradation 1 \
    --weight_clipping 0.1 \
    --test \
    --world_settings ernestogym/envs/single_agent/ijcnn_deg_test.yaml \
    --electrical_model ernestogym/ernesto/data/battery/models/electrical/thevenin_pack.yaml \
    --aging_model ernestogym/ernesto/data/battery/models/aging/bolun_pack.yaml;

python launch_env.py \
    --algo only_market \
    --n_cores 5 \
    --exp_name "ijcnn_presentation" \
    --spread_factor 1 \
    --weight_trading 1 \
    --weight_operational_cost 0 \
    --weight_degradation 1 \
    --weight_clipping 0.1 \
    --test \
    --world_settings ernestogym/envs/single_agent/ijcnn_deg_test.yaml \
    --electrical_model ernestogym/ernesto/data/battery/models/electrical/thevenin_pack.yaml \
    --aging_model ernestogym/ernesto/data/battery/models/aging/bolun_pack.yaml;
