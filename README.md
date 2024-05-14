# GenReL-World
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/CIMBIBOY/GenRel-MetaWorld/blob/master/LICENSE)

# The present state of GenReL-World is a work in progress. In case of constructive toughts please open an issue.

__Table of Contents__
- [Abstract](#abstract)
- [Project Work](#project-work)
- [Setup](#setup)
- [Frameworks](#frameworks)
  * [MetaWorld](#metaworld)
  * [MuJoCo](#mujoco)
  * [MuJoCo Menagerie](#mujoco-menagerie)
  * [TorchRL](#torchrl)
  * [CleanRL](#cleanrl)
  * [Gymnasium](#gymnasium)
  * [Basic PPO](#basic-ppo)
- [Usage and Training](#usage-and-training)
- [Evaluation](#evaluation)
- [Improvements](#improvements)
- [References](#references)
- [Credits](#credits)

## Abstract

__GenReL-World is a general Reinforcement Learning framework to utilize various world models as environments for robot manipulation.__

The goal is to contrust the framework which utilizes general reinforcement learning algorithms to teach and control a robot in it's given environment for it's given task. 
A big problem in robotics is, that agents adapt hardly to new environments or tasks, which fall out of the trained task distribution. 
The adaptation of a general framework can be worthwhile if they can be trained on different world models (latent representations) and environments, tested on hyperparameters and initial conditions and using different reward and action space constructions. 

Implementing different algorithms and finding connections between them is an ongoing research area, which can play a crutial part in robotics.
Also data structures, exploration functions and the definition of state, reward and action spaces can impact the learning of an agent.
The research focuses on different intitial values, action and rewards spaces and the change of agents behaviour in different environments.

The framework involves reinforcement learning concepts from [(0)](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf) and meta-reinforcement learning and multi-task learning using the MetaWorld open-source simulated benchmark [(1)](https://meta-world.github.io/). 

The project utilizes Google DeepMinds's MuJoCo (Multi-Joint dynamics with Contact) as a general purpose physics engine that aims to facilitate research and development in robotics. [(2)](https://mujoco.org/). 

The project also includes a built 7 degree of freedom robotic arm which is simulated with MuJoCo Menagerie's xArm7 (or some other model) as a part of the collection of high-quality models for the MuJoCo physics engine, curated by Google DeepMind [(3)](https://github.com/google-deepmind/mujoco_menagerie) [(4)](https://github.com/google-deepmind/mujoco_menagerie/tree/main/ufactory_xarm7). 

Different algorithm objectives have been consider as the open community provides better and better solutions and implementations, such as torchrl and cleanrl and also Basic [PPO](https://github.com/ericyangyu/PPO-for-Beginners) algorithms.

I also share the dream of Yann LeCun about how to construct autonomous intelligent agents [(5)](https://openreview.net/pdf?id=BZ5a1r-kVsf).

## Project Work 

__The project is conducted by Mark Czimber__

The bases of GenReL-World is part of the 2024 Deep Learning class of Aquincum Institute of Technology project work. 
The project is based on various frameworks and algorithms to understand general reinforcement learning behaviour, to later implement Model Predictive Control or integrate World Models. Training various environments with basic agents to understand the differences between different parametrizations and initial values.  

Agents are trained on simpler tasks such as MoonLander and also in complex envrionments such as MetaWorld. Taught to land or control a robotic arm in a virtual environment. 
This can later be scaled by the mixing of different algorithms and implementations, hopefully even new approaches. 

Several ongoing reasearch papers are taken into account during the development of the project [(5)](https://openreview.net/pdf?id=BZ5a1r-kVsf) [(6)](https://arxiv.org/abs/2301.08028) [(7)](https://arxiv.org/abs/2010.02193) [(8)](https://github.com/ericyangyu/PPO-for-Beginners).  

## Setup

GenReL-World is based on [python3](https://www.python.org/downloads/) high-level language, which is widely used for reinforcement learning. 
The project also requires several python library dependencies which can be installed from [PyPI](https://pypi.org/) using the pip install "required library" terminal command. 

The most important libraries for the project, which needs to be installed is poetry whcih is a version control package

First steps:
```
git clone https://github.com/CIMBIBOY/GenReL-World.git
cd GenReL-World 
pip install poetry
poetry shell 
```

Configure the python interpreter to ('.venv': Poetry, by adding path to Enter Interpreter Path section, usually at "parent_folder_path"/GenReL-World/.venv/bin/python3
Installation steps after shell is active: 
```
poetry update
pip install -r requirements.txt
```

## Frameworks
Here is a list of github repositories of the used sources: 

* [MetaWorld](https://meta-world.github.io/) is an open-source simulated benchmark for meta-reinforcement learning and multi-task learning. 
* [MuJoCo](https://github.com/google-deepmind/mujoco) is a general purpose physics engine that aims to facilitate research and development in robotics, biomechanics, graphics and animation, machine learning, and other areas. 
* [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) is a collection of high-quality models for the MuJoCo physics engine including models that work well right out of the gate.
* [TorchRL](https://github.com/pytorch/rl) is an open-source Reinforcement Learning (RL) library for PyTorch.
* [Cleanrl](https://github.com/vwxyzjn/cleanrl) is a Deep Reinforcement Learning library based on the Gymnasium API that provides high-quality single-file implementation with research-friendly features. 
* [Gymnasium](https://gymnasium.farama.org/) is an open source Python library for developing and comparing reinforcement learning algorithms by providing a standard API to communicate between learning algorithms and environments.
* [BasicPPO](https://github.com/ericyangyu/PPO-for-Beginners) is Eric Yu's help to beginners to get started in writing Proximal Policy Optimization (PPO) from scratch using PyTorch. 

### MetaWorld
A visualization of MetaWorld can be done in MuJoCo's 3D environment with running:
```
mjpython test_env/testMetaW.py
```
The script intagrates MetaWorld's pick-place-v2-goal-observable into MuJoCo for visualization. 

### MuJoCo
To install MuJoCo follow installation steps on the [MuJoCo github](https://github.com/google-deepmind/mujoco) page. 
For python users this is a simple installation from PyPI as pip install mujaco, already added by poetry. 
MuJoCo can also be downloaded from the [offical site](https://mujoco.org/).

### MuJoCo Menagerie
To install MuJoCo Menagerie follow the installation steps on the [MuJoCo Menagerie github](https://github.com/google-deepmind/mujoco_menagerie/tree/main?tab=readme-ov-file#installation-and-usage). 

To visalize xArm7 run:
```
mjpython test_env/testxArm7.py
```

The script intagrates xArm7 xml into MuJoCo for visualization. 

The xarm7.xml can also be dragged to MuJoCo app downloaded from the official cite.

### TorchRL
Is the implementation go-to if coding from scratch, it also comes with a set of highly re-usable functionals for cost functions, returns and data processing.

### Cleanrl
Clean and simple implementations of RL algorithms, yet scaleable to run thousands of experiments using AWS Batch.

### Gymnasium
[Gymnasium](https://github.com/Farama-Foundation/Gymnasium) is OpenAI's Gym's future maintance which is an elegant and very useful libary for RL environments.

### Basic PPO
Huge thanks to the author, is a clean and easily understandable code, I ran quite a few trains with different environments and hyperparameters on the PPO!

## Usage and Training
In case of any errors due to sys path definition please redefine your location as path variable.

Different rewards can be defined for MetaWorld Pick-Place-v2 in GenReL-World/metaworld/envs/mujoco/sawyer_xyz/v2/sawyer_pick_place_v2.py
Currently there is a MetaWorld and a slef logic impmeneted compute_reward function.

__To train algorithms on different environments run:__

From scratch PPO on MoonLanderContinuous-v2:
"Under correction"
```
python train/traingymPPO.py
```

From scratch PPO on MetaWorld Pick-Place-v2:
"Under correction"
```
mjpython train/trainMetaWPPO.py
```

Basic PPO on MoonLanderContinuous-v2::
"Good evaluation"
To train from scratch:
```
python PPO_gym/main.py
```

To test best model from saved weights: 
```
python PPO_gym/main.py --mode test --actor_model ppo_actor.pth
```

To train best model from saved weights: 
```
python PPO_gym/main.py --actor_model ppo_actor.pth --critic_model ppo_critic.pth
```

Modified Basic PPO on MetaWorld Pick-Place-v2:
"Struggle because of task complexity"
To train from scratch:
```
mjpython PPO_metaW/main.py
```

To test best model from saved weights: 
```
mjpython PPO_metaW/main.py --mode test --actor_model ppo_actor.pth
```

To train best model from saved weights: 
"MetaWorld predefined reward"
```
mjpython PPO_metaW/main.py --actor_model ppo_meta_actor.pth --critic_model ppo_meta_critic.pth
```

"Own reward system"
```
mjpython PPO_metaW/main.py --actor_model ppo_meta_actor_v4.pth --critic_model ppo_meta_critic_v4.pth
```

Garage PPO on MetaWorld Pick-Place-v2:
"Currently fails"
```
mjpython train/train_garagePPOv2.py
```

TorchRL PPO on MetaWorld Pick-Place-v2:
"Currently fails"
```
mjpython train/train_torchrl.py
```

## Evaluation
Evaluation is done by testing learned policy for environments and Weights and Biases monitoring. Own Wandb config can be defined in each models PPO Agent init class, and also use of logging can be switched on-of by the use of wand_use=True or False setting.

Wandb graphs and visualizations can be found in the [documentation](docs)

## Improvements 
The project focuses on hyperparameters and initial settings of simple RL models, which allows different setting to be tried. 
Trains are conducted on a Mac M2 chip, which is not the fastest GPU to train with. 
Improvement can be done by fixing the scratch PPO implementation and further hyperparamter optimalization.
Utilizing torchrl's, cleanrl's and garage's models can provide more information of agent's behavior in different world/environment settings. 
Model Predictive Control can provide a better understanding of the latent embedding of the environment.
Parametrized or infinite action spaces can be considered, which is one of my research area. 
Different Agents can be implemented for better performance. 
Testing of various reward fucntions can further improve model performance. 

## References 

[(0)](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf) Richard S. Sutton and Andrew G. Barto. (2018). ReinforcementLearning: An Introduction (second edition). The MIT Press. 

(1) [Official cite](https://meta-world.github.io/) and
[Github by Farama-Foundation](https://github.com/Farama-Foundation/Metaworld)
MetaWorld: 
```
@inproceedings{yu2019meta,
  title={Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning},
  author={Tianhe Yu and Deirdre Quillen and Zhanpeng He and Ryan Julian and Karol Hausman and Chelsea Finn and Sergey Levine},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2019}
  eprint={1910.10897},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
  url={https://arxiv.org/abs/1910.10897}
}
```

(2) [Official cite](https://mujoco.org/) and
[Github](https://github.com/google-deepmind/mujoco)
MuJoCo: 
```
@inproceedings{todorov2012mujoco,
  title={MuJoCo: A physics engine for model-based control},
  author={Todorov, Emanuel and Erez, Tom and Tassa, Yuval},
  booktitle={2012 IEEE/RSJ International Conference on Intelligent Robots and Systems},
  pages={5026--5033},
  year={2012},
  organization={IEEE},
  doi={10.1109/IROS.2012.6386109}
}
```

(3) [Github](https://github.com/google-deepmind/mujoco_menagerie)
MuJoCo Menagerie:
```
@software{menagerie2022github,
  author = {Zakka, Kevin and Tassa, Yuval and {MuJoCo Menagerie Contributors}},
  title = {{MuJoCo Menagerie: A collection of high-quality simulation models for MuJoCo}},
  url = {http://github.com/google-deepmind/mujoco_menagerie},
  year = {2022},
}
```

[(4)](https://github.com/google-deepmind/mujoco_menagerie/tree/main/ufactory_xarm7) MuJoCo Menagerie xArm7.

[(5)](https://openreview.net/pdf?id=BZ5a1r-kVsf) Yann LeCun. (2022). A Path Towards Autonomous Machine Intelligence, Version 0.9.2, 2022-06-27. Courant Institute of Mathematical Sciences, New York University and Meta - Fundamental AI Research. 

[(6)](https://arxiv.org/abs/2301.08028) Jacob Beck et al. (2023). A Survey of Meta-Reinforcement Learning. arXiv:2301.08028 [cs.LG].

[(7)](https://arxiv.org/abs/2010.02193) Danijar Hafner et al. (2020).	Mastering Atari with Discrete World Models. arXiv:2010.02193 [cs.LG].

[(8)](https://github.com/ericyangyu/PPO-for-Beginners) Eric Yang Yu, (2023). PPO for Beginners.

## Credits 

Huge thanks and big credit to the [Meta AI](https://ai.meta.com/) and [Google Deepmind](https://deepmind.google/) who are one of the most influential intelligence laboratories and builders of the open community research!








