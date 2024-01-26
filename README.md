# Generative Model of Policies: Exploring the Latent Space with Human Feedback

![Python Version](https://img.shields.io/badge/Python->=3.10-blue)
![Code Style](https://img.shields.io/badge/Code_Style-black-black)

[**Installation**](#installation) | [**Overview**](#overview) 

## Installation

This project requires Python 3.10 or later and a working [JAX](https://github.com/google/jax) installation.
To install JAX, refer to [the instructions](https://github.com/google/jax#installation).

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Overview

There are three main scripts. Each have a number of command line arguments that can be obtained by running: `python <script_name>.py --help`.

### Training

To run a training, use the `train.py` script. This will create a folder in the directory `results/` which contains a `config` file. By the end of the training a `tasks.png` visualization should also automatically be created. See `--help` for more information on the hyperparameters.

### Human Feedback

To optimize in the latent space with human feedback, run the `humanfeedback.py` script. You can precise the run folder with `--run_path` or the environment with `--env`. See `--help` for more information.

At the end, a `pathhf.npy` should be created, as well as a plot representing the path inside the latent space.

### Interpolation between behaviors

To linearly interpolate between behaviors, run the `interpolation.py` script. This will directly fetch the `successes.npz` file created after training the agent, calculate the barycenters of each task in the latent space and start the visualization. You can move the slider to move between behaviors.
