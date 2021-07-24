import torch
from torch.optim import Adam
import gym
import time
import models as core


class VPGBuffer:
    """
    A buffer for storing trajectories and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """