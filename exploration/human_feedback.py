import numpy as np

from gmp.latent_space import within_norm


class HumanFeedback:
    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def next_latents(self, n: int, prev_latent: np.ndarray) -> tuple[np.ndarray, ...]:
        alpha = self.rng.uniform(0.001, 0.05, size=(n, 1))
        latents = within_norm(
            prev_latent + alpha * self.rng.standard_normal((n, prev_latent.shape[-1])),
            1.0,
        )
        return tuple([np.expand_dims(l, axis=0) for l in latents])
