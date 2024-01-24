import os

import jax
import matplotlib.pyplot as plt
import numpy as np
import flatdict

from rl.callbacks.wandb_callback import WandbCallback
import rl.config as cfg

from envs import make_vec_env, make_vec_task_env

from gmp.config import GmpParams
from gmp.gmp_closefar import GMP
from gmp.latent_space import random_ball_numpy

from test_vecenv import rollout_vec_env


def make_algo_config_env(seed: int, config: dict) -> cfg.AlgoConfig:
    envs, env_cfg = make_vec_env(config["task"], seed + 1, 8)

    n_env_steps = 500_000 // 8 if config["task"] == "cartpole" else 100_000 // 8

    return (
        cfg.AlgoConfig(
            seed,
            GmpParams(
                gamma=0.99,
                _lambda=0.95,
                clip_eps=0.2,
                entropy_coef=0.01,
                value_coef=0.5,
                normalize=True,
                latent_size=config["latent_size"],
                diversity_latent_samples=config["diversity_latent_samples"],
                latent_coef=config["latent_coef"],
                hidden_size=64,
                activation_fn="relu",
                m_hidden_size=config["m_hidden_size"],
                m_activation_fn="relu",
                m_n_layers=config["m_n_layers"],
                architecture=config["architecture"],
                n_blocks=config["n_blocks"],
            ),
            cfg.UpdateConfig(
                learning_rate=0.0001,
                learning_rate_annealing=True,
                max_grad_norm=0.5,
                max_buffer_size=256,
                batch_size=256,
                n_epochs=5,
                shared_encoder=False,
            ),
            cfg.TrainConfig(n_env_steps=n_env_steps, save_frequency=n_env_steps),
            env_cfg,
        ),
        envs,
    )


def init_train_eval(seed: int, config: dict):
    gmpconfig, envs = make_algo_config_env(seed, config)
    dimension = gmpconfig.algo_params.latent_size

    gmp = GMP(gmpconfig)
    gmp.train(
        envs,
        gmpconfig.train_cfg.n_env_steps,
        callbacks=[
            WandbCallback(project="sweep_gmp", entity="raffael", name=gmp.run_name)
        ],
    )
    del envs

    envs, tasks = make_vec_task_env(config["task"], seed + 2, 100)
    points = np.load(f"./points_{dimension}d.npy")
    # points = random_ball_numpy(np.random.default_rng(0), 100, dimension)
    successes = rollout_vec_env(seed + 3, gmp, envs, points, tasks)

    np.savez(os.path.join("./results", gmp.run_name, "successes.npz"), **successes)

    latent_idx = np.array(successes["latent_idx"])
    colors = ["r", "b", "g", "y"]
    plt.clf()
    for i, t in enumerate(tasks):
        values = points[latent_idx[np.where(successes[t])]]
        if dimension == 2:
            plt.scatter(values[..., 0], values[..., 1], c=colors[i], label=t)
        elif dimension == 1:
            plt.scatter(
                values[..., 0], np.zeros_like(values[..., 0]), c=colors[i], label=t
            )
    plt.legend()
    plt.savefig(os.path.join("./results", gmp.run_name, "tasks"))


def configs(sweep_config):
    import itertools

    keys, values = zip(*sweep_config.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    all_dicts = []
    for d in permutations_dicts:
        arch = d["architecture"]
        for n in sweep_config["architecture"][arch]:
            all_dicts.append(d | {"n_blocks": n})

    return all_dicts


def main():
    sweep_config = {
        "architecture": {"StyleAdaIN": [1, 2], "Multiplicative": [0]},
        "diversity_latent_samples": [4, 8],
        "latent_coef": [0.0, 0.05, 0.1, 0.2],
        "latent_size": [2],
        "m_hidden_size": [16, 64],
        "m_n_layers": [0, 4, 8],
        "task": ["cartpole", "ring"],
    }
    all_configs = configs(sweep_config)
    for c in all_configs:
        init_train_eval(0, c)


if __name__ == "__main__":
    main()
