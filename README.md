# ErNESTO-gym

This repository contains the implementation of the **ErNESTO-gym**, a Reinforcement Learning
testbed for smart grids.
The framework is grounded on the [ErNESTO digital twin](https://github.com/Daveonwave/DT-rse/tree/master) which provides a realistic model of a battery system, and is designed to tackle control problem in the field of energy storage employing RL techniques.

This project comes from the collaboration of [Politecnico di Milano](https://www.polimi.it) and [RSE](https://www.rse-web.it).

## Folder structure

```
docs/                       # website and documentation
examples/                   # example code for running the environment
ernestogym/                 # main Python package
    ernesto/
        energy_storage/     # simulator of the energy storage
        data/               # exogenous data of the environment
        ./                  # demand, generation and market objects
    envs/
        {env}/              # Gymnasium environment of the micro grid
    algorithms/
        ./                  # algorithms already implemented within the framework
scripts/
    *.sh                    # scripts to run experiments
launch_env.py               # entry point of the environment
```

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

To reproduce the experiments, a `bash` file contained in the `scripts/` folder must be run from the root directory. For example:

```bash
./scripts/baseline.sh
```

Edit the bash files to choose different configuration files or models. The possible options can be retrieved by running `python launch_env.py --help`.
Notice that `yaml` configuration files, contained in `ernestogym/ernesto/data/battery/` folder, have to adhere to a standard formatting, validated within the script [schema.py](./ernestogym/ernesto/preprocessing/schema.py).
Follow the formatting of the already provided configuration file to generate new ones.

#### Results visualization

Experiment results can be visualized in the [notebooks](./examples/single_agent/).

## :triangular_flag_on_post: Roadmap

The idea is to extend the framework to handle a more complex scenario, considering not only a single energy
storage system, but a broader Smart Grid. To reach this goal, further steps have to be taken. In particular:

1. Extend the research to a **multi-agent RL (MARL)** setting
2. Corroborate the simulator with new models and allow more configuration options

[comment]: <> (### Examples)

## :paperclip: Citing

```

```
