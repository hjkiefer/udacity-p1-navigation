# Project 1: Navigation

## Introduction

This project is about training and agent to collect bananas. The environment is
a square world, where yellow and blue bananas are to be collected. 

The agent is rewards with a +1 reward for collecting yellow bananas, and a -1
reward for collecting a blue banana. As the agent tries to maximise the
cumulative reward, it should collect yellow bananas, and avoid blue bananas.

The environment state is a preprocessed view with is consist of the agent's
velocity and ray-based perception around the agents forward direction. This
state is prepared by the environment, and we are thus not working with the
pixel view that we (humans) are seeing.

The agent have 4 actions available:

- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

Each episode consist of a defined number of actions, and the agent must get an
average score of +13 over 100 consecutive episodes in order to solve the
environment.

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
