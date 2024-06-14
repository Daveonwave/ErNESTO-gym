python launch_env.py \
    --algo ppo \
    --n_envs 4 \
    --exp_name "no_weights_norm_trad" \
    --n_episodes 5 \
    --weight_trading 1 \
    --weight_operational_cost 1 \
    --weight_clipping 1 \
    --train 