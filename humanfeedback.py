from pathlib import Path

from absl import app, flags
import matplotlib.pyplot as plt
import numpy as np

from envs import make_task_env
from evaluation.human_feedback import HumanFeedback, loop, create_pygame_screen
from evaluation.plots import plot_tasks_successes
from gmp.gmp import GMP


flags.DEFINE_integer("seed", 0, "Seed for reproducibility.")
flags.DEFINE_string("run_path", "", "Path of the run folder.")
FLAGS = flags.FLAGS


def main(_):
    gmp = GMP.unserialize(FLAGS.run_path)
    gmp.restore()
    env_name = gmp.config.env_cfg.task_name

    env0, _ = make_task_env(env_name, FLAGS.seed, render_mode="rgb_array")
    env1, _ = make_task_env(env_name, FLAGS.seed + 1, render_mode="rgb_array")
    screen = create_pygame_screen(env0)

    start_latent = np.zeros((2,))
    HF = HumanFeedback(FLAGS.seed + 2)

    path = loop(screen, start_latent, env0, env1, gmp, "Reach left border", HF)
    np.save(Path(FLAGS.run_path).joinpath("pathhf.npy"), np.array(path))

    successes_path = Path(FLAGS.run_path).joinpath("successes.npz")
    if successes_path.exists():
        successes = np.load(successes_path)
        plot_tasks_successes(
            successes,
            show_plot=False,
            clear_figure=False,
            color_fn=lambda i: ["r", "b", "g"][i],
        )

    plt.plot(
        np.array(path)[..., 0],
        np.array(path)[..., 1],
        c="black",
        linewidth=3,
        label="Human Feedback path",
    )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    app.run(main)
