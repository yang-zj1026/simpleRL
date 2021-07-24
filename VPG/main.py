<<<<<<< HEAD
import numpy as np
import torch
from torch.optim import Adam
import gym
import utils as core
=======
import torch
from torch.optim import Adam
import gym
import time
import models as core
>>>>>>> 0abda3f85654a2574a8a5bb599ac4d8ad34873fc


class VPGBuffer:
    """
    A buffer for storing trajectories and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
<<<<<<< HEAD
    """

    def __init__(self, obs_dim, action_dim, size, gamma=0.99, lam=0.95):
        self.obs_buffer = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.action_buffer = np.zeros(core.combined_shape(size, action_dim), dtype=np.float32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.logp_buffer = np.zeros(size, dtype=np.float32)
        self.gamma = gamma
        self.lam = lam
        self.ptr = 0  # current index
        self.path_start_index = 0  # path start index
        self.max_size = size  # max length of buffer

    def store(self, obs, action, reward, value, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # make sure buffer still has room
        self.obs_buffer[self.ptr] = obs
        self.action_buffer[self.ptr] = action
        self.reward_buffer[self.ptr] = reward
        self.value_buffer[self.ptr] = value
        self.logp_buffer[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_value):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        path_slice = slice(self.path_start_index, self.ptr)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        # GAE-Lambda advantage calculation, read GAE paper for more info
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.advantage_buffer[path_slice] = core.discount_sum(deltas, self.gamma * self.lam)

        # rewards-to-go calculation
        self.return_buffer[path_slice] = core.discount_sum(rewards, self.gamma)[:-1]

        self.path_start_index = self.ptr

    def get_data(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size
        # reset buffer variable
        self.ptr = 0
        self.path_start_index = 0

        # calculate the mean and std of advantage buffer
        adv_mean, adv_std = np.mean(self.advantage_buffer), np.std(self.advantage_buffer)
        self.advantage_buffer = (self.advantage_buffer - adv_mean) / adv_std
        data = dict(obs=self.obs_buffer, act=self.action_buffer, ret=self.return_buffer,
                    adv=self.advantage_buffer, logp=self.logp_buffer)
        return {k: torch.tensor(v, dtype=torch.float32) for k, v in data.items()}


class VPG:
    def __init__(self, env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=None, seed=0, steps_per_epoch=4000,
                 epoch=50, gamma=0.99, pi_lr=3e-4, value_lr=1e-3, train_v_iters=80, lam=0.97, max_ep_len=1000,
                 log_interval=50):
        self.env_fn = env_fn
        self.actor_critic = actor_critic
        self.ac = None
        self.ac_kwargs = ac_kwargs
        self.seed = seed
        self.steps_per_epoch = steps_per_epoch
        self.epoch = epoch
        self.gamma = gamma
        self.pi_lr = pi_lr
        self.value_lr = value_lr
        self.train_v_iters = train_v_iters
        self.lam = lam
        self.max_ep_len = max_ep_len
        self.log_interval = log_interval

    def compute_loss_pi(self, data):
        obs, action, adv = data['obs'], data['act'], data['adv']

        # Policy loss and Value loss
        pi, logp = self.ac.pi(obs, action)
        loss_pi = -(logp * adv).mean()
        return loss_pi

    def compute_loss_v(self, data):
        obs = data['obs']
        ret = data['ret']
        return ((self.ac.v(obs) - ret) ** 2).mean()

    def update(self, buf, pi_optimizer, value_optimizer):
        data = buf.get_data()
        loss_pi_old = self.compute_loss_pi(data).item()
        loss_v_old = self.compute_loss_v(data).item()

        # Train policy with a single step of gradient descent
        pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Value function learning
        for i in range(self.train_v_iters):
            value_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data)
            loss_v.backward()
            value_optimizer.step()

        return loss_pi_old, loss_v_old, loss_pi.item(), loss_v.item()

    def train(self):
        torch.manual_seed(self.seed)

        # make environment
        env = self.env_fn()
        obs_dim = env.observation_space.shape
        action_dim = env.action_space.shape

        # Create actor-critic module
        self.ac = self.actor_critic(env.observation_space, env.action_space)

        buffer = VPGBuffer(obs_dim=obs_dim, action_dim=action_dim, size=self.steps_per_epoch,
                           gamma=self.gamma, lam=self.lam)

        pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.pi_lr)
        value_optimizer = Adam(self.ac.v.parameters(), lr=self.value_lr)

        for epoch in range(self.epoch):
            obs, ep_return, ep_len, num_eps = env.reset(), 0, 0, 0
            for t in range(self.steps_per_epoch):
                action, value, logp = self.ac.step(torch.tensor(obs, dtype=torch.float32))
                next_obs, reward, done, _ = env.step(action)

                ep_return += reward
                ep_len += 1
                obs = next_obs

                buffer.store(obs, action, reward, value, logp)

                timeout = (ep_len == self.max_ep_len)
                terminal = (done or timeout)
                epoch_end = (t == self.steps_per_epoch - 1)

                if terminal or epoch_end:
                    # if epoch_end and not terminal:
                    # print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if timeout or epoch_end:
                        _, v, _ = self.ac.step(torch.tensor(obs, dtype=torch.float32))
                    else:
                        v = 0
                    buffer.finish_path(v)
                    num_eps += 1
                    obs, ep_ret, ep_len = env.reset(), 0, 0

            loss_pi_old, loss_v_old, loss_pi, loss_v = self.update(buf=buffer, pi_optimizer=pi_optimizer,
                                                                   value_optimizer=value_optimizer)

            if epoch % self.log_interval == 0:
                print("Epoch: %3d, Reward: %.4f, Policy Loss: %.4f, Value Loss: %.4f "
                      % (epoch, ep_return / num_eps, loss_pi_old, loss_v_old))
            if epoch == self.epoch - 1:
                print("Epoch: %3d, Reward: %.4f, Policy Loss: %.4f, Value Loss: %.4f "
                      % (epoch, ep_return / num_eps, loss_pi_old, loss_v_old))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--epoch', type=int, default=100)
    args = parser.parse_args()

    vpg = VPG(lambda: gym.make(args.env), gamma=args.gamma, seed=args.seed,
              steps_per_epoch=args.steps, epoch=args.epoch, log_interval=10)

    vpg.train()
=======
    """
>>>>>>> 0abda3f85654a2574a8a5bb599ac4d8ad34873fc
