python launch_env.py \
    --algo a2c \
    --n_cores 5 \
    --exp_name "rev1_market_ask_noise" \
    --load_model "a2c_10500000_steps.zip" \
    --weight_trading 1 \
    --weight_operational_cost 1 \
    --weight_clipping 0.1 \
    --test 