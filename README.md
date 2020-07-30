# UdacityRL-Tennis

![Tennis](/images/agents.gif)

# Project Details

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The state of an agent looks like: [ 0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.         -6.65278625 -1.5
 -0.          0.          6.83172083  6.         -0.          0.        ]

The task is episodic, and in order to solve the environment, agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents).

# Getting Started

* create (and activate) a new Python environment. I'd recommend using python version 3.6 since it's compatible with all other project's dependencies.

* install unityagents with:
`pip install unityagents`

* install [pytorch](https://pytorch.org/)

* Download the Unity Environment that matches your operating system:

Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

* in case if you're getting troubles wiht the dependencies setup please take a look at the dependencies section from the [Udacity DRL course repository](https://github.com/udacity/deep-reinforcement-learning/blob/master/README.md)

# Instructions

For running agent training algorithm please run scripts/main.py script.
Please note that you might need to modify scripts/main.py script and set a proper path to the unity environment (by default env_path = "../Tennis_Windows_x86_64/Tennis.exe").


