from dataclasses import dataclass, field
from typing import Callable

import flax.linen as nn
import rl.config as cfg
from rl.algos.ppo import PPOParams


@dataclass
class GmpParams(PPOParams):
    latent_size: int
    diversity_latent_samples: int
    latent_coef: float

    hidden_size: int
    activation_fn: str

    # mapping
    m_hidden_size: int
    m_activation_fn: str
    m_n_layers: int


if __name__ == "__name__":
    p = GmpParams()
