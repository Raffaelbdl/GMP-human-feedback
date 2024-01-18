import jax

import rl.config as cfg

from gmp.gmp import GmpParams
from gmp.module import train_state_factory

from tests.test_env import make_test_env


def test_train_state_factory():
    """Checks if the train state if successfully created.s"""
    _, env_cfg = make_test_env()
    config = cfg.AlgoConfig(
        0,
        GmpParams(latent_size=2),
        cfg.UpdateConfig(1e-3, False, 0.5, 256, 128, 3, False),
        cfg.TrainConfig(10**5, -1),
        env_cfg,
    )
    train_state = train_state_factory(jax.random.key(0), config, tabulate=True)
    assert True
