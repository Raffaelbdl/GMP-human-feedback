from absl import app, flags

import rl.config as cfg

from envs import make_env

from gmp.config import GmpParams
from gmp.gmp import GMP


flags.DEFINE_integer("seed", 0, "Seed for reproducibility.")
flags.DEFINE_enum(
    "task", "cartpole", ["cartpole", "ring"], "Name of the task.", short_name="t"
)
flags.DEFINE_string("run_name", None, "Name of the run.")
# Diversity loss
flags.DEFINE_integer("latent_size", 2, "Dimension of the latent space.")
flags.DEFINE_integer(
    "diversity_latent_samples", 8, "Number of samples for the diversity loss term."
)
flags.DEFINE_float("latent_coef", 0.1, "Coefficient in front of the loss term.")
# Encoder
flags.DEFINE_integer("hidden_size", 64, "Size of the Dense layers inside the encoder.")
flags.DEFINE_enum(
    "activation_fn", "tanh", ["tanh", "relu"], "Activation function inside the encoder."
)
# Mapping network
flags.DEFINE_integer(
    "m_hidden_size", 64, "Size of the Dense layers inside the mapping network."
)
flags.DEFINE_enum(
    "m_activation_fn",
    "tanh",
    ["tanh", "relu"],
    "Activation function inside the mapping network.",
)
flags.DEFINE_integer("m_n_layers", 2, "Number of layers inside the mapping network.")
FLAGS = flags.FLAGS


def make_config(seed: int, env_cfg: cfg.EnvConfig) -> cfg.AlgoConfig:
    return cfg.AlgoConfig(
        seed,
        GmpParams(
            gamma=0.99,
            _lambda=0.95,
            clip_eps=0.2,
            entropy_coef=0.01,
            value_coef=0.5,
            normalize=True,
            latent_size=FLAGS.latent_size,
            diversity_latent_samples=FLAGS.diversity_latent_samples,
            latent_coef=FLAGS.latent_coef,
            hidden_size=FLAGS.hidden_size,
            activation_fn=FLAGS.activation_fn,
            m_hidden_size=FLAGS.m_hidden_size,
            m_activation_fn=FLAGS.m_activation_fn,
            m_n_layers=FLAGS.m_n_layers,
            architecture="StyleAdaIN",
            n_blocks=2,
        ),
        cfg.UpdateConfig(
            learning_rate=0.0003,
            learning_rate_annealing=True,
            max_grad_norm=0.5,
            max_buffer_size=256,
            batch_size=128,
            n_epochs=1,
            shared_encoder=False,
        ),
        cfg.TrainConfig(n_env_steps=10**5, save_frequency=10**5),
        env_cfg,
    )


def main(_):
    env, env_cfg = make_env(FLAGS.task, FLAGS.seed)
    algo_cfg = make_config(FLAGS.seed, env_cfg)

    gmp = GMP(algo_cfg, run_name=FLAGS.run_name, tabulate=True)
    gmp.train(env, algo_cfg.train_cfg.n_env_steps, callbacks=[])


if __name__ == "__main__":
    app.run(main)
