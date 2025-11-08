from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.patch_gym import _patch_env
from typing import Any, Callable, Dict, List, Optional, Type, Union
import gymnasium as gym
import os

def make_vec_env_custom(
    env_id: Union[str, Callable[..., gym.Env]],
    n_envs: int = 1,
    seed: Optional[Union[int, List[int]]] = None,
    start_index: int = 0,
    monitor_dir: Optional[str] = None,
    wrapper_class: Optional[Callable[[gym.Env], gym.Env]] = None,
    env_kwargs: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    vec_env_cls: Optional[Type[Union[DummyVecEnv, SubprocVecEnv]]] = None,
    vec_env_kwargs: Optional[Dict[str, Any]] = None,
    monitor_kwargs: Optional[Dict[str, Any]] = None,
    wrapper_kwargs: Optional[Dict[str, Any]] = None,
    render_mode: Optional[str] = None,
):
    """
    Custom version of make_vec_env that supports:
    - Passing a list of env_kwargs (one per environment)
    - Passing a list of seeds (one per environment)
    - Gymnasium compatibility
    """

    vec_env_kwargs = vec_env_kwargs or {}
    monitor_kwargs = monitor_kwargs or {}
    wrapper_kwargs = wrapper_kwargs or {}

    # Handle env_kwargs (dict or list)
    if isinstance(env_kwargs, dict):
        env_kwargs_list = [env_kwargs] * n_envs
    elif isinstance(env_kwargs, list):
        assert len(env_kwargs) == n_envs, (
            f"Expected env_kwargs list of length {n_envs}, got {len(env_kwargs)}"
        )
        env_kwargs_list = env_kwargs
    else:
        env_kwargs_list = [{} for _ in range(n_envs)]

    # Handle seeds (int or list)
    if isinstance(seed, int):
        seed_list = [seed + i for i in range(n_envs)]
    elif isinstance(seed, list):
        assert len(seed) == n_envs, (
            f"Expected seed list of length {n_envs}, got {len(seed)}"
        )
        seed_list = seed
    else:
        seed_list = [None] * n_envs

    def make_env(rank: int) -> Callable[[], gym.Env]:
        def _init() -> gym.Env:
            kwargs = env_kwargs_list[rank]
            if render_mode:
                kwargs["render_mode"] = render_mode

            # Create environment
            if isinstance(env_id, str):
                env = gym.make(env_id, **kwargs)
            else:
                env = env_id(**kwargs)
                env = _patch_env(env)

            # Proper seeding
            env_seed = seed_list[rank]
            env.reset(seed=env_seed)

            # Monitor setup
            monitor_path = (
                os.path.join(monitor_dir, str(rank)) if monitor_dir else None
            )
            if monitor_path:
                os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(env, filename=monitor_path, **monitor_kwargs)

            # Apply wrapper if provided
            if wrapper_class:
                env = wrapper_class(env, **wrapper_kwargs)

            return env

        return _init

    if vec_env_cls is None:
        vec_env_cls = DummyVecEnv

    return vec_env_cls([make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs)