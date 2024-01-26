import time

import pygame


def wait_sync(duration: float):
    start = time.time()
    while time.time() - start < duration:
        pass
    return True


def create_pygame_screen(
    width: int,
    height: int,
    window_title: str,
    *,
    bot_offset: int,
    vert_mid_offset: int,
    top_offset: int
) -> pygame.Surface:
    """Instantiates a pygame screen."""
    pygame.init()
    pygame.font.init()

    pygame.display.set_caption(window_title)
    screen = pygame.display.set_mode(
        (width + vert_mid_offset, height + top_offset + bot_offset)
    )

    return screen
