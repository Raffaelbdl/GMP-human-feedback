from enum import Enum
import time

import cv2
import gymnasium as gym
import numpy as np
import pygame

from gmp.latent_space import within_norm
from gmp.gmp import GMP

FPS = 60
TOP_OFFSET = 40
BOT_OFFSET = 40
MID_OFFSET = 2


class State(Enum):
    SELECT = 0
    ROLLOUT = 1


class HumanFeedback:
    """Human Feedback class.

    Holds information about the temperature coefficient,
        and handles sampling new latent vectors.

    Attributes:
        alpha: a float
        decrease_coef: a float that regulate alpha decrease
        increase_coef: a float that regulate alpha increase
    """

    def __init__(
        self, seed: int, *, decrease_coef: float = 0.8, increase_coef: float = 1.1
    ) -> None:
        """Instantiates a HumanFeedback instance.

        Args:
            seed: an int for reproducibility.
        """
        self.rng = np.random.default_rng(seed)
        self.alpha = 0.1
        self.decrease_coef = decrease_coef
        self.increase_coef = increase_coef

    def decrease_alpha(self) -> None:
        self.alpha *= self.decrease_coef

    def increase_alpha(self) -> None:
        self.alpha *= self.increase_coef

    def next_latents(self, n: int, prev_latent: np.ndarray) -> np.ndarray:
        latents = prev_latent
        latents += self.alpha * self.rng.standard_normal((n, prev_latent.shape[-1]))
        latents = within_norm(latents, 1.0)
        return latents

    def update_alpha(self, selected: bool) -> None:
        """Increases or decreases alpha based on selection.

        If new latent selected, decrease alpha.
        If previous latent kept, increase alpha.
        """
        if selected:
            self.decrease_alpha()
        else:
            self.increase_alpha()


def create_pygame_screen(env: gym.Env) -> pygame.Surface:
    """Instantiates a pygame screen for the human-feedback interface."""
    pygame.init()
    pygame.font.init()

    pygame.display.set_caption("Human Adaptable Policies")
    env_shape = env.render().shape
    screen = pygame.display.set_mode(
        (2 * env_shape[1] + MID_OFFSET, env_shape[0] + TOP_OFFSET + BOT_OFFSET)
    )

    return screen


def surface_title(title: str) -> pygame.Surface:
    font = pygame.font.SysFont("Comic Sans MS", 30)
    return font.render(title, False, (255, 255, 255))


def surface_end() -> pygame.Surface:
    font = pygame.font.SysFont("Comic Sans MS", 30)
    return font.render("Left, None, Right", False, (255, 255, 255))


def draw_frames(f0: np.ndarray, f1: np.ndarray, screen: pygame.Surface) -> None:
    middle = np.zeros((f0.shape[0], MID_OFFSET, f0.shape[2]))
    fcat = np.concatenate([f0, middle, f1], axis=1)
    fcat = cv2.rotate(fcat, cv2.ROTATE_90_CLOCKWISE)
    fcat = cv2.flip(fcat, 1)
    fcat = pygame.surfarray.make_surface(fcat)
    screen.blit(fcat, (0, TOP_OFFSET))


def rollout(
    agent: GMP,
    env0: gym.Env,
    env1: gym.Env,
    prev_latent: np.ndarray,
    screen: pygame.Surface,
    human_feedback: HumanFeedback,
) -> tuple[np.ndarray, np.ndarray]:
    """Performs one rollout in two environments and
    draws the resulting frames."""
    # hide bottom text
    size = screen.get_size()
    screen.blit(
        pygame.surfarray.make_surface(np.zeros((size[0], BOT_OFFSET, 3))),
        (0, size[1] - BOT_OFFSET),
    )

    o0, _ = env0.reset()
    t0 = False

    o1, _ = env1.reset()
    t1 = False

    l0, l1 = human_feedback.next_latents(2, prev_latent)

    def step_env(
        env: gym.Env, observation: np.ndarray, latent: np.ndarray, agent: GMP
    ) -> tuple[np.ndarray, bool, np.ndarray]:
        action, _ = agent.skip_explore(
            {
                "observation": np.expand_dims(observation, axis=0),
                "latent": np.expand_dims(latent, axis=0),
            }
        )
        observation, _, done, trunc, _ = env.step(action[0])
        frame = env.render()
        terminated = done or trunc
        return frame, terminated, observation

    while not t0 or not t1:
        start_time = time.time()

        if not t0:
            f0, t0, o0 = step_env(env0, o0, l0, agent)
        if not t1:
            f1, t1, o1 = step_env(env1, o1, l1, agent)

        draw_frames(f0, f1, screen)
        pygame.display.update()
        while time.time() - start_time < 1 / FPS:
            pass

    screen.blit(surface_end(), (0, screen.get_size()[1] - BOT_OFFSET))
    return l0, l1


def loop(
    screen: pygame.Surface,
    prev_latent: np.ndarray,
    env0: gym.Env,
    env1: gym.Env,
    agent: GMP,
    task: str,
    human_feedback: HumanFeedback,
) -> list[np.ndarray]:
    """Launches the human feedback interface.

    Args:
        screen: a pygame Surface.
        prev_latent: the starting latent as an Array.
        env0: a gymnasium environment.
        env1: a gymnasium environment.
        agent: a GMP agent.
        task: the alternative task the human should help
            optimize as a string.
        human_feedback: a HumanFeedback instance.

    Returns:
        The path of latent vectors selected during the optimization.
    """
    state = State.ROLLOUT

    title = surface_title(task)
    screen.blit(title, (0, 0))

    l0, l1 = prev_latent, prev_latent

    latent_path = [prev_latent]

    active = True
    while active:
        if state == State.ROLLOUT:
            l0, l1 = rollout(agent, env0, env1, prev_latent, screen, human_feedback)
            state = State.SELECT

        elif state == State.SELECT:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    active = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        prev_latent = l0
                        latent_path.append(l0)
                        human_feedback.decrease_alpha()
                        state = State.ROLLOUT
                    elif event.key == pygame.K_RIGHT:
                        prev_latent = l1
                        latent_path.append(l1)
                        human_feedback.decrease_alpha()
                        state = State.ROLLOUT
                    elif event.key == pygame.K_DOWN:
                        state = State.ROLLOUT
                        human_feedback.increase_alpha()

        pygame.display.update()

    return latent_path
