from pathlib import Path

from absl import app, flags

import gymnasium as gym
import numpy as np
import rl.config as cfg

from envs import make_vec_task_env, make_vec_env
from evaluation.alternative_tasks import rollout_vec_env
from evaluation.plots import plot_tasks_successes

from gmp.config import GmpParams
from gmp.gmp import GMP
from gmp.latent_space import random_ball_numpy

flags.DEFINE_integer("seed", 0, "Seed for reproducibility.")
flags.DEFINE_enum("task", "cartpole", ["cartpole", "ring"], "Task name.")

flags.DEFINE_integer(
    "diversity_latent_samples", 8, "Number of samples for the diversity loss term."
)
flags.DEFINE_float("latent_coef", 0.05, "Coefficient in front of the loss term.")

flags.DEFINE_integer("hidden_size", 64, "Size of the Dense layers inside the encoder.")
flags.DEFINE_enum(
    "activation_fn", "relu", ["tanh", "relu"], "Activation function inside the encoder."
)

flags.DEFINE_integer(
    "m_hidden_size", 16, "Size of the Dense layers inside the mapping network."
)
flags.DEFINE_enum(
    "m_activation_fn",
    "relu",
    ["tanh", "relu"],
    "Activation function inside the mapping network.",
)
flags.DEFINE_integer("m_n_layers", 4, "Number of layers inside the mapping network.")

flags.DEFINE_enum(
    "architecture",
    "style",
    ["style", "multiplicative"],
    "Architecture of the generator.",
)
flags.DEFINE_integer("n_blocks", 1, "Number of blocks in the style architecture.")

FLAGS = flags.FLAGS


def make_config(
    seed: int,
    task: str,
    *,
    diversity_latent_samples: int,
    latent_coef: float,
    hidden_size: int,
    activation_fn: str,
    m_hidden_size: int,
    m_activation_fn: str,
    m_n_layers: int,
    architecture: str,
    n_blocks: int,
    n_envs: int = 8,
) -> tuple[gym.vector.VectorEnv, cfg.AlgoConfig]:
    n_envs = max(2, n_envs)
    envs, env_cfg = make_vec_env(task, seed, n_envs)
    n_env_steps = 500_000 // n_envs if task == "cartpole" else 100_000 // 8

    gmp_params = GmpParams(
        gamma=0.99,
        _lambda=0.95,
        clip_eps=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        normalize=True,
        latent_size=2,
        diversity_latent_samples=diversity_latent_samples,
        latent_coef=latent_coef,
        hidden_size=hidden_size,
        activation_fn=activation_fn,
        m_hidden_size=m_hidden_size,
        m_activation_fn=m_activation_fn,
        m_n_layers=m_n_layers,
        architecture=architecture,
        n_blocks=n_blocks,
    )
    config = cfg.AlgoConfig(
        seed,
        gmp_params,
        cfg.UpdateConfig(
            learning_rate=0.0001,
            learning_rate_annealing=True,
            max_grad_norm=0.5,
            max_buffer_size=256,
            batch_size=max(32, 256 * n_envs // 8),
            n_epochs=5,
            shared_encoder=False,
        ),
        cfg.TrainConfig(n_env_steps=n_env_steps, save_frequency=n_env_steps),
        env_cfg,
    )

    return config, envs


def eval_alt_tasks(
    seed: int,
    gmp: GMP,
    envs: gym.Env | gym.vector.VectorEnv,
    tasks: list[str],
    points: np.ndarray,
):
    successes = rollout_vec_env(seed + 1, gmp, envs, points, tasks)
    np.savez(Path("./results").joinpath(gmp.run_name, "successes.npz"), **successes)
    plot_tasks_successes(
        successes,
        show_plot=True,
        save_path=Path("./results").joinpath(gmp.run_name, "tasks.png"),
        color_fn=lambda i: ["r", "b", "g"][i],
    )


def train_and_eval_alt_tasks(
    seed: int,
    envs: gym.Env | gym.vector.VectorEnv,
    config: cfg.AlgoConfig,
    points: np.ndarray,
):
    gmp = GMP(config)
    gmp.train(envs, config.train_cfg.n_env_steps, [])
    del envs

    envs, tasks = make_vec_task_env(config.env_cfg.task_name, seed, 100)
    eval_alt_tasks(seed + 2, gmp, envs, tasks, points)


def main(_):
    config, envs = make_config(
        FLAGS.seed,
        FLAGS.task,
        diversity_latent_samples=FLAGS.diversity_latent_samples,
        latent_coef=FLAGS.latent_coef,
        hidden_size=FLAGS.hidden_size,
        activation_fn=FLAGS.activation_fn,
        m_hidden_size=FLAGS.m_hidden_size,
        m_activation_fn=FLAGS.m_activation_fn,
        m_n_layers=FLAGS.m_n_layers,
        architecture=FLAGS.architecture,
        n_blocks=FLAGS.n_blocks,
    )
    points = random_ball_numpy(np.random.default_rng(FLAGS.seed + 1), 10_000, 2)
    train_and_eval_alt_tasks(FLAGS.seed + 2, envs, config, points)


if __name__ == "__main__":
    app.run(main)
