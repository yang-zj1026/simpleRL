import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box


# Build a feedforward neural network.
def mlp(size, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for j in range(len(size) - 1):
        act_func = activation if j < len(size) - 2 else output_activation
        layers += [nn.Linear(size[j], size[j + 1]), act_func()]
    return nn.Sequential(*layers)


# The training process
def train(env_name='CartPoleNoFramskip-v4', hidden_sizes=32, lr=1e-2,
          epochs=50, batch_size=5000):
    # make environment, check type of spaces, get obs / act dims
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    acts_dim = env.action_space.n

    # build the policy network
    logits_net = mlp(size=[obs_dim] + [hidden_sizes] + [acts_dim])

    # compute action distribution
    def get_policy(obs):
        policy_logits = logits_net(obs)
        return Categorical(logits=policy_logits)

    # action selection function (outputs int actions, sampled from policy)
    def get_action(obs):
        return get_policy(obs).sample().item()

    # make loss function whose gradient is policy gradient
    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    # initialize the optimizer
    optimizer = Adam(logits_net.parameters(), lr=lr)

    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []        # for observations
        batch_acts = []       # for actions
        batch_weights = []    # for return R(tau) weighting in policy gradient
        batch_returns = []    # for episode returns
        batch_lens = []       # for episode lengths
        batch_loss = 0

        # reset episode-specific variables
        obs = env.reset()   # first obs comes from starting distribution
        done = False        # signal from environment suggesting that episode is over
        ep_rewards = []     # list for rewards accrued throughout ep

        # collect experience by acting in the environment with current policy
        while True:
            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            act = get_action(torch.tensor(obs, dtype=torch.float32))
            obs, reward, done, _ = env.step(act)

            # save action and reward
            batch_acts.append(act)
            ep_rewards.append(reward)

            if done:
                # if episode is over, record info about episode
                ep_return, ep_len = sum(ep_rewards), len(ep_rewards)
                batch_returns.append(ep_return)
                batch_lens.append(ep_len)

                # the weight for each logprob
                batch_weights += [ep_return] * ep_len

                # reset episode-specific variables
                obs, done, ep_rewards = env.reset(), False, []

                if len(batch_obs) > batch_size:
                    break

        # policy gradient update in each epoch
        optimizer.zero_grad()
        batch_loss = compute_loss(obs=torch.tensor(batch_obs, dtype=torch.float32),
                                  act=torch.tensor(batch_acts, dtype=torch.int32),
                                  weights=torch.tensor(batch_weights, dtype=torch.float32))
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_returns, batch_lens

    # training loop
    for i in range(epochs):
        loss, batch_returns, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f' %
              (i, loss, np.mean(batch_returns), np.mean(batch_lens)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing simplest formulation of policy gradient.\n')
    train(env_name=args.env_name, lr=args.lr, batch_size=5000)




