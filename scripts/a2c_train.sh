python launch_env.py \
    --algo a2c \
    --n_envs 4 \
    --gamma 0.9 \
    --exp_name "rev3_final_noclip" \
    --n_episodes 5 \
    --weight_trading 1 \
    --weight_operational_cost 1 \
    --weight_clipping 0 \
    --train 