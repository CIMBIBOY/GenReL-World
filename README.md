# GenReL-World
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/CIMBIBOY/GenRel-MetaWorld/blob/master/LICENSE)

# In the present GenReL-World is a work in progress. In case of constructive toughts please open an issue.

__Table of Contents__
- [Abstract](#abstract)
- [Project Work](#project)
- [Setup](#setup)
- [Installation of Bases](#installation-of-bases)
  * [MetaWorld](#metaworld)
  * [MuJoCo](#mujoco)
  * [MuJoCo Menagerie](#menagerie)
- [References](#references)
- [Credits](#credits)

## Abstract

__GenReL-World is a general Reinforcement Learning framework to utilize various world models as environments for robot manipulation.__

The goal is to contrust the framework witch utilizes general reinforcement learning algorithms to control a robotic arm. A big problem in robotics is, that they adapt hardly to new environments or to task, which fall out to the trained task distribution. 
The adaptation of a general framework can be worthwhile if they can be intagrated to be trained on different world models. 

With that only the world model and the encoding to a lower latent representation have to be switched. 
Implementing the different algorithms and finding connections between them is an ongoing research area, which can play a crutial part in robotics. 

The framework involves reinforcement learning concepts from [(0)](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf) and meta-reinforcement learning and multi-task learning using the MetaWorld open-source simulated benchmark [(1)](https://meta-world.github.io/). 

The project utilizes Google DeepMinds's MuJoCo (Multi-Joint dynamics with Contact) as a general purpose physics engine that aims to facilitate research and development in robotics. [(2)](https://mujoco.org/). The project also includes a built 7 degree of freedom robotic arm which is simulated with MuJoCo Menagerie's xArm7 (or some other model) as a part of the collection of high-quality models for the MuJoCo physics engine, curated by Google DeepMind [(3)](https://github.com/google-deepmind/mujoco_menagerie) [(4)](https://github.com/google-deepmind/mujoco_menagerie/tree/main/ufactory_xarm7). 

We also share the dream of Yann LeCun about how to construct autonomous intelligent agents [(5)](https://openreview.net/pdf?id=BZ5a1r-kVsf).

## Project Work 

__The project is conducted by Mark Czimber and Josh Kang__

The bases of GenReL-World is part of the 2024 Deep Learning class of Aquincum Institute of Technology project work. 
The first milestones include simple implementations of reinforcement learning algorithms for example along with meta reinforcement learning. 

These will be used to simpler task such as moving objects and controlling a robotic arm in a virtual environment. 
This can later be scaled by the mixing of different algorithms and implementations of new approaches. 

Several ongoing reasearch papers are taken into account during the development of the project [(5)](https://openreview.net/pdf?id=BZ5a1r-kVsf) [(6)](https://arxiv.org/abs/2301.08028) [(7)](https://arxiv.org/abs/2010.02193).  

## Setup

GenReL-World is based on [python3](https://www.python.org/downloads/) high-level language, which is widely used for reinforcement learning. 
The project also requires several python library dependencies which can be installed from [PyPI](https://pypi.org/) using the pip install "required library" terminal command. 

The most important libraries for the project, which needs to be installed are: metaworld, mujaco, torch, gymnasium, scipy and numpy.

## Installation of Bases
Here is a list of github repositories of the used sources: 

* [MetaWorld](https://meta-world.github.io/) is an open-source simulated benchmark for meta-reinforcement learning and multi-task learning. 
* [MuJoCo](https://github.com/google-deepmind/mujoco) is a general purpose physics engine that aims to facilitate research and development in robotics, biomechanics, graphics and animation, machine learning, and other areas. 
* [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) is a collection of high-quality models for the MuJoCo physics engine including models that work well right out of the gate.

### MetaWorld
To install MetaWorld follow the [installation steps]([https://meta-world.github.io/](https://github.com/Farama-Foundation/Metaworld?tab=readme-ov-file#installation).
The README of MetaWorld is a worthwhile read which can be found [here](https://github.com/Farama-Foundation/Metaworld/blob/master/README.md).

A visualization of MetaWorld can be done in MuJoCo's 3D environment with running [testMetaW.py](). 
It is advised to create the testMetaW.py in the instalaltion folder of MetaWorld: /path/to//metaworld here create testMetaW.py. 

The python script can be run with mjpython from a terminal window by finding the path to mujoco instalaltion: /path/to/mujoco/bin/mjpython. 
If the file does not have the execute permission, it can be added by running: chmod +x /path/to/testMetaW.py. 

To run use: mjpython /path/to/testMetaW.py. The script intagrates MetaWorld's ML1 into MuJoCo for visualization. 

### MuJoCo
To install MuJoCo follow installation steps on the [MuJoCo github](https://github.com/google-deepmind/mujoco) page. 
For python users this is a simple installation from PyPI as pip install mujaco. 
MuJoCo can also be downloaded from the [offical site](https://mujoco.org/).

### MuJoCo Menagerie
To install MuJoCo Menagerie follow the installation steps on the [MuJoCo Menagerie github](https://github.com/google-deepmind/mujoco_menagerie/tree/main?tab=readme-ov-file#installation-and-usage). To visalize xArm7 run the [testxArm7.py](). 
It is advised to create the testxArm7.py in the instalaltion folder of MetaWorld: /path/to//metaworld here create testxArm7.py.

The path to xArm7's xml file have to be specified in testxArm7.py in line: 
mujoco.MjModel.from_xml_path('/path/to/mujoco_menagerie/ufactory_xarm7/xarm7.xml'). 

The python script can be run with mjpython from a terminal window by finding the path to mujoco instalaltion: /path/to/mujoco/bin/mjpython. If the file does not have the execute permission, it can be added by running: chmod +x /path/to/testxArm7.py. 

To run use: mjpython /path/to/testxArm7.py. The script intagrates xArm7 xml into MuJoCo for visualization. 

The xarm7.xml can also be dragged to MuJoCo app downloaded from the official cite.

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

## Credits 

Huge thanks and big credit to the [Meta AI](https://ai.meta.com/) and [Google Deepmind](https://deepmind.google/) who are one of the most influential intelligence laboratories and builders of the open community research!








