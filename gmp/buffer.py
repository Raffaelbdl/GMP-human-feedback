from collections import namedtuple

ExperienceLatent = namedtuple(
    "ExperienceLatent",
    [
        "observation",
        "latent",
        "action",
        "reward",
        "done",
        "next_observation",
        "log_prob",
    ],
)
