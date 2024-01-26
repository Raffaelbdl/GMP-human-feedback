import time

import cv2
import gymnasium as gym
import numpy as np
import pygame
import pygame_widgets
from pygame_widgets.slider import Slider

from evaluation.pygame_utils import wait_sync
from gmp.gmp import GMP

BOT_OFFSET = 40


class LatentInterpolation:
    """Latent Interpolation class.

    Handles linear interpolation between positions
        connected by segments.

    Attributes:
        checkpoints: the list of positions to interpolate between.
    """

    def __init__(self, checkpoints: list[np.ndarray]):
        self.checkpoints = checkpoints
        self.positions = np.linspace(0, 1, len(self.checkpoints))

    def lerp(self, a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        return t * (b - a) + a

    def get_latent(self, t: float) -> np.ndarray:
        """Returns the latent corresponding to a ratio of the total path.

        Args:
            t: a float between 0 and 1 that corresponds to the ratio of the
                total path.

        Returns:
            The corresponding latent vector as an Array.
        """
        a, o = self.checkpoints[0], 0.0
        for c, p in zip(self.checkpoints[1:], self.positions[1:]):
            if t <= p:
                return self.lerp(a, c, (t - o) / (p - o))
            a, o = c, p
        return -1


def create_pygame_screen(env: gym.Env) -> pygame.Surface:
    """Instantiates a pygame screen for the interpolation interface."""
    from evaluation.pygame_utils import create_pygame_screen as screen_fn

    shape = env.render().shape
    return screen_fn(
        shape[1],
        shape[0],
        "Human Adaptable Policies",
        bot_offset=BOT_OFFSET,
        vert_mid_offset=0,
        top_offset=0,
    )


def draw_frame(frame: np.ndarray, screen: pygame.Surface) -> None:
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    frame = cv2.flip(frame, 1)
    frame = pygame.surfarray.make_surface(frame)
    screen.blit(frame, (0, 0))


def step_env(
    agent: GMP,
    env: gym.Env,
    obs: np.ndarray,
    latent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Performs a single step in env."""
    action = agent.skip_explore(
        {
            "observation": np.expand_dims(obs, axis=0),
            "latent": np.expand_dims(latent, axis=0),
        }
    )[0]
    obs, _, done, trunc, _ = env.step(action[0])
    frame = env.render()
    if done or trunc:
        obs, _ = env.reset()
    return obs, frame


def loop(
    screen: pygame.Surface,
    agent: GMP,
    env: gym.Env,
    latent_interpolation: LatentInterpolation,
    *,
    fps: int = 25,
):
    """Launches the interpolation interface.

    Args:
        screen: a pygame Surface.
        prev_latent: the starting latent as an Array.
        env: a gymnasium environment.
        slider: a Slider object.
        latent_interpolation: a LatentInterpolation instance.
    """
    slider = Slider(
        screen,
        0,
        screen.get_height() - BOT_OFFSET,
        screen.get_width(),
        BOT_OFFSET,
        min=0.0,
        max=1.0,
        step=0.05,
    )
    slider_position = 0.0
    slider.setValue(slider_position)

    obs, _ = env.reset()
    frame = env.render()
    latent = latent_interpolation.get_latent(slider_position)

    active = True
    while active:
        obs, frame = step_env(agent, env, obs, latent)
        draw_frame(frame, screen)

        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                active = False
                quit()

        pygame_widgets.update(events)

        new_slider_position = slider.getValue()
        if new_slider_position != slider_position:
            slider_position = new_slider_position
            latent = latent_interpolation.get_latent(slider_position)

        pygame.display.update()

        wait_sync(1 / fps)
