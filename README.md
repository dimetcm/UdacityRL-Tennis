# UdacityRL-ContinuousControl

![Reacher](/images/agents.gif)

# Project Details

The project shows an example of an agent controlling double-jointed arm that learns to reach target location. The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. The goal of the  agent is to maintain its position at the target location for as many time steps as possible. Environment contains 20 identical agents, each with its own copy of the environment. The environment is considered solved when agents get an average score of +30 (over 100 consecutive episodes, and over all agents).

# Getting Started

* create (and activate) a new Python environment. I'd recommend using python version 3.6 since it's compatible with all other project's dependencies.

* install unityagents with:
`pip install unityagents`

* install [pytorch](https://pytorch.org/)

* Download the Unity Environment that matches your operating system:

Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

* in case if you're getting troubles wiht the dependencies setup please take a look at the dependencies section from the [Udacity DRL course repository](https://github.com/udacity/deep-reinforcement-learning/blob/master/README.md)

# Instructions

For running agent training algorithm please run scripts/main.py script.
Please note that you might need to modify scripts/main.py script and set a proper path to the unity environment (by default env_path = "../data/MA/Reacher.exe").


