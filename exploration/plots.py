import matplotlib.pyplot as plt
import numpy as np

from exploration.hulls import Delaunay, Hull1D, in_hull
from gmp.latent_space import random_ball_numpy


def plot_values1d(points, values, vmin=-1, vmax=1):
    p = plt.scatter(points, np.zeros_like(points), c=values, cmap="spring")
    plt.colorbar(p)
    plt.clim(vmin, vmax)

    plt.xlim((-1.0, 1.0))
    plt.ylim((-1.0, 1.0))

    plt.gca().set_aspect("equal")

    plt.show()


def plot_values2d(points, values, vmin=-1, vmax=1):
    p = plt.scatter(points[..., 0], points[..., 1], c=values, cmap="spring")
    plt.colorbar(p)
    plt.clim(vmin, vmax)

    plt.xlim((-1.0, 1.0))
    plt.ylim((-1.0, 1.0))

    plt.gca().set_aspect("equal")

    plt.show()


def plot_convex_hulls2d(classes: dict[str, np.ndarray]):
    rng = np.random.default_rng(0)
    cmap = plt.cm.get_cmap("hsv", 32)
    colors = rng.integers(0, 32, size=len(classes))

    for i, (task, points) in enumerate(classes.items()):
        if len(points) == 0:
            continue

        hull = Delaunay(points)
        new_points = []
        for p in random_ball_numpy(rng, 10_000, 2, 1.0):
            if in_hull(p, hull):
                new_points.append(p)
        new_points = np.array(new_points)
        plt.scatter(
            new_points[..., 0], new_points[..., 1], color=cmap(colors[i]), label=task
        )

    plt.legend()

    plt.xlim((-1.0, 1.0))
    plt.ylim((-1.0, 1.0))

    plt.gca().set_aspect("equal")

    plt.show()


def plot_convex_hulls1d(return_pts, left_pts, right_pts):
    return_hull = Hull1D(return_pts) if len(return_pts) > 0 else None
    left_hull = Hull1D(left_pts) if len(left_pts) > 0 else None
    right_hull = Hull1D(right_pts) if len(right_pts) > 0 else None

    new_pts = random_ball_numpy(100_000, 1, 1)
    nreturn_pts, nleft_pts, nright_pts = [], [], []
    for p in new_pts:
        if left_hull and in_hull(p, left_hull):
            nleft_pts.append(p)
        elif right_hull and in_hull(p, right_hull):
            nright_pts.append(p)
        elif return_hull and in_hull(p, return_hull):
            nreturn_pts.append(p)
    nreturn_pts = np.array(nreturn_pts)
    nleft_pts = np.array(nleft_pts)
    nright_pts = np.array(nright_pts)

    if return_hull and len(return_pts) > 0:
        plt.scatter(
            nreturn_pts[..., 0],
            np.zeros_like(nreturn_pts[..., 0]),
            color="b",
            label="high return",
        )
    if left_hull and len(nleft_pts) > 0:
        plt.scatter(
            nleft_pts[..., 0], np.zeros_like(nleft_pts[..., 0]), color="r", label="left"
        )
    if right_hull and len(nright_pts) > 0:
        plt.scatter(
            nright_pts[..., 0],
            np.zeros_like(nright_pts[..., 0]),
            color="g",
            label="right",
        )
    plt.legend()

    plt.xlim((-1.0, 1.0))
    plt.ylim((-1.0, 1.0))

    plt.gca().set_aspect("equal")

    plt.show()
