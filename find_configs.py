import os
from pathlib import Path
import yaml


def find_all_config_with(
    paths: list[str], configs: list[dict], task: str = None, **key_values
):
    valid_paths = []
    valid_configs = []
    for p, c in zip(paths, configs):
        if task:
            if c["task_name"] != task:
                continue
        for k, v in key_values.items():
            if c["algo_params"][k] != v:
                break
        else:
            valid_paths.append(p)
            valid_configs.append(c)
    return valid_paths, valid_configs


def load_all_configs(dir: Path):
    all_paths = []
    all_configs = []
    for p in os.listdir(dir):
        path = dir.joinpath(p, "config")
        all_paths.append("./" + str(dir.joinpath(p, "tasks.png")))
        all_configs.append(yaml.load(path.open("r"), yaml.SafeLoader))
    return all_paths, all_configs


def main():
    paths, configs = load_all_configs(Path("./results/"))
    print(
        find_all_config_with(
            paths,
            configs,
            # task="CartPole-v1",
            task="Ring",
            # architecture="Multiplicative",
            architecture="StyleAdaIN",
            diversity_latent_samples=8,
            latent_coef=0.2,
            m_hidden_size=16,
            m_n_layers=8,
            n_blocks=2,
        )[0]
    )


if __name__ == "__main__":
    main()
