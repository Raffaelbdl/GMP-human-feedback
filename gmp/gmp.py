from dataclasses import dataclass, field
from typing import Callable

import flax.linen as nn
import rl.config as cfg


@dataclass
class MappingParams(cfg.AlgoParams):
    hidden_size: int = 64
    activation_fn: Callable = nn.relu
    n_layers: int = 2


@dataclass
class GmpParams(cfg.AlgoParams):
    latent_size: int

    mapping: MappingParams = field(default_factory=lambda: MappingParams())
    hidden_size: int = 64
    activation_fn: Callable = nn.relu
