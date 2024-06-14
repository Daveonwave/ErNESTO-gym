python launch_env.py \
    --algo a2c \
    --n_cores 5 \
    --exp_name "no_weights_norm_trad" \
    --load_model "a2c_10500000_steps.zip" \
    --weight_trading 1 \
    --weight_operational_cost 1 \
    --weight_clipping 1 \
    --test 