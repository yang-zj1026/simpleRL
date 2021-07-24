import torch
import torch.nn as nn
import numpy as np
import scipy.signal
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from gym.spaces import Box, Discrete


# Build a feedforward neural network.
def mlp(size, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for j in range(len(size) - 1):
        act_func = activation if j < len(size) - 2 else output_activation
        layers += [nn.Linear(size[j], size[j + 1]), act_func()]
    return nn.Sequential(*layers)


# Combine scalar and tuple
def combined_shape(length, shape=None):
    if shape is None:
        return length,
    return (length, shape) if np.isscalar(shape) else (length, *shape)


# calculate the discounted sum
def discount_sum(x, gamma):
    """
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-gamma)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):
    def _distribution(self, obs):
        return NotImplementedError

    def _log_prob_from_distribution(self, pi, action):
        raise NotImplementedError

    def forward(self, obs, action=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_action = None
        if action is not None:
            logp_action = self._log_prob_from_distribution(pi, action)
        return pi, logp_action


# Policy network for environment with discrete action space, see PG/simple.
class MLPCategoricalActor(Actor):
    def __init__(self, obs_dim, action_dim, hidden_size, activation):
        super().__init__()
        self.policy_net = mlp([obs_dim] + list(hidden_size) + [action_dim], activation)

    def _distribution(self, obs):
        logits = self.policy_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, action):
        return pi.log_prob(action)


# Policy network for environment with continuous action space
class MLPGaussianActor(Actor):
    def __init__(self, obs_dim, action_dim, hidden_size, activation):
        super().__init__()
        log_std = -0.5 * np.ones(action_dim, dtype=np.float32)
        self.log_std = nn.Parameter(torch.tensor(log_std))
        self.policy_net = mlp([obs_dim] + list(hidden_size) + [action_dim], activation)

    def _distribution(self, obs):
        mu = self.policy_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, action):
        return pi.log_prob(action).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution


# Value Network
class MLPValueCritic(nn.Module):
    def __init__(self, obs_dim, hidden_size, activation):
        super().__init__()
        self.v_net = mlp([obs_dim]+list(hidden_size) + [1], activation=activation)

    def forward(self, obs):
        # ensure v has right shape
        return torch.squeeze(self.v_net(obs), -1)


# Actor combines policy network and value network
class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()
        obs_dim = observation_space.shape[0]

        # build policy actor depending on the type of action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v = MLPValueCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]
