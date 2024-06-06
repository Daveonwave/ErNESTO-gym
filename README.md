# ErNESTO-gym

This repository contains the implementation of the **ErNESTO-gym**, a Reinforcement Learning
testbed for smart grids.
The framework is grounded on the [ErNESTO digital twin](https://github.com/Daveonwave/DT-rse/tree/master) which provides a realistic model of a battery system, and is designed to tackle control problem in the field of energy storage employing RL techniques.

This project comes from the collaboration of [Politecnico di Milano](https://www.polimi.it) and [RSE](https://www.rse-web.it).

## :hammer_and_wrench: Installation

In order to use this codebase you need to work with a Python version >= 3.11.
To use ErNESTO-gym, clone this repository and install the required libraries:

```bash
git clone https://github.com/Daveonwave/ErNESTO-gym.git && \
cd ErNESTO-gym/ && \
python -m pip install -r requirements.txt
```

## :brain: Usage

Before launching any script, add to the PYTHONPATH the root folder `ErNESTO-gym/`:

```bash
export PYTHONPATH=$(pwd)
```

#### Reproducibility

To reproduce the experiments, a `bash` file must be run in the main folder. For example:

```bash
python launch_env.py
```

Notice that `yaml` configuration files, contained in `ernestogym/ernesto/data/battery/` folder, have to adhere to a standard formatting, validated within the script [schema.py](./ernestogym/ernesto/preprocessing/schema.py).
Follow the formatting of the already provided configuration file to generate new ones.

#### Results visualization

Experiment results can be visualized in the jupyter notebooks files.

## :triangular_flag_on_post: Roadmap

The idea is to extend the framework to handle a more complex scenario, considering not only a single energy
storage system, but a broader Smart Grid. To reach this goal, further steps have to be taken. In particular:

1. Extend the research to a **multi-agent RL (MARL)** setting
2. Corroborate the simulator with new models and allow more configuration options

[comment]: <> (### Examples)

## :paperclip: Citing

```

```
