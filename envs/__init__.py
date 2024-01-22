import gymnasium as gym
import rl.config as cfg


def make_env(task: str, seed: int) -> tuple[gym.Env, cfg.EnvConfig]:
    if task == "cartpole":
        from envs.cartpole import make_cartpole

        return make_cartpole(seed=seed)
    elif task == "ring":
        from envs.ring import make_ring

        return make_ring(seed=seed)
    else:
        raise NotImplementedError


def make_vec_env(task: str, seed: int, n_envs: int) -> tuple[gym.Env, cfg.EnvConfig]:
    if task == "cartpole":
        from envs.cartpole import make_vec_cartpole

        return make_vec_cartpole(seed=seed, n_envs=n_envs)
    elif task == "ring":
        from envs.ring import make_vec_ring

        return make_vec_ring(seed=seed, n_envs=n_envs)
    else:
        raise NotImplementedError


def make_task_env(
    task: str, seed: int, render_mode: str | None = None
) -> tuple[gym.Env, list[str]]:
    if task == "cartpole":
        from envs.cartpole import make_task_cartpole

        return make_task_cartpole(seed=seed, render_mode=render_mode)
    elif task == "ring":
        from envs.ring import make_task_ring

        return make_task_ring(seed=seed, render_mode=render_mode)
    else:
        raise NotImplementedError


def make_vec_task_env(
    task: str, seed: int, n_envs: int
) -> tuple[gym.Env, cfg.EnvConfig]:
    if task == "cartpole":
        from envs.cartpole import make_vec_task_cartpole

        return make_vec_task_cartpole(seed=seed, n_envs=n_envs)
    elif task == "ring":
        from envs.ring import make_vec_task_ring

        return make_vec_task_ring(seed=seed, n_envs=n_envs)
    else:
        raise NotImplementedError
