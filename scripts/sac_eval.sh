python launch_env.py \
    --algo sac \
    --n_cores 5 \
    --exp_name "rev3_final" \
    --load_model "sac_2000000_steps.zip" \
    --weight_trading 1 \
    --weight_operational_cost 1 \
    --weight_clipping 0.1 \
    --test 