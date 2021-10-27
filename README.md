# Project 1: Navigation

## Introduction

In this project, an agent was trained to collect bananas, in a square world.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1
is provided for collecting a blue banana.  Thus, the goal of the agent is to
collect as many yellow bananas as possible while avoiding blue bananas. 

The state space has 37 dimensions and contains the agent's velocity, along with
ray-based perception of objects around agent's forward direction.  Given this
information, the agent has to learn how to best select actions.  Four discrete
actions are available, corresponding to:

- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get
an average score of +13 over 100 consecutive episodes.

## Setting up your environment

It has not been possible to setup the environment locally due to setup issues
with old versions of the unity environment, so for me, it was necessary to run it in the
Udacity provided cloud VM which has all dependencies already installed. 

Optionally, you can attempt to install the dependencies yourself by following the guide here
[udacity drl github repo](https://github.com/udacity/deep-reinforcement-learning#dependencies)

## Training the agent

Run the code in the [Navigation.ipynb](Navigation.ipynb) to train the agent.
This loads models from [model.py](model.py) and loads the agent from
[dqn_agent.py](dqn_agent.py) and finally runs a training loop which end swhen
the environment has been solved.
