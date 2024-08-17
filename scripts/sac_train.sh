python launch_env.py \
    --algo sac \
    --n_envs 4 \
    --gamma 0.9 \
    --exp_name "rev3_final" \
    --n_episodes 1 \
    --weight_trading 1 \
    --weight_operational_cost 1 \
    --weight_clipping 0.1 \
    --train 