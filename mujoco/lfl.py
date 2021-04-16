"""Reconstructs a reward function from learning trajectories.

This code is an adaptation of the code of LfL for MuJoCo taken from https://github.com/alexis-jacq/Learning_from_a_Learner

Here is the implementation of the algorithm described
in Learning from a Learner (LfL)
(http://proceedings.mlr.press/v97/jacq19a/jacq19a.pdf).
"""

import glob
import os
from arguments import get_args
from gym.spaces.box import Box
from model import Policy
import numpy as np
import torch
import torch.nn as nn


args = get_args()

args.demos_dir = args.demos_dir
args.save_dir = args.save_dir
args.scores_dir = args.scores_dir

try:
  os.makedirs(args.rewards_dir)
except OSError:
  files = glob.glob(os.path.join(args.rewards_dir, '*'+args.expe+'.pth'))
#   for file in files:
#     os.remove(file)

try:
  os.makedirs(args.policies_dir)
except OSError:
  files = glob.glob(os.path.join(args.policies_dir, '*'+args.expe+'.pth'))
  # for file in files:
  #   os.remove(file)


# trajectory points used in the paper for each environment:
#----------------------------------------------------------
if args.env_name == 'Reacher-v2':
  observer_steps = list(range(10,20))
if args.env_name == 'Hopper-v2':
  observer_steps = list(range(10,20))


# observer_steps = list(range(0, 3))

# hyperparameters used in the paper:
#-----------------------------------
# entropy_coef = 0.01
entropy_coef = 0
gamma = 0.99
n_demos_per_update = 2000


demos_dir = args.demos_dir
demos_expe_dir = args.demos_dir + '/expe_'+ str(args.expe) + '_' + str(args.env_name)


n_epoch_policies = 1000
n_epoch_kl = 100
n_epoch_reward_init = 3000
n_epoch_reward_shaping = 1000


lr_policies = 1e-3
lr_kl = 1e-3
lr_reward_init = 1e-3
lr_reward_shaping = 1e-3

# just to get env dimmensions:
actions_file_name_0 = demos_expe_dir + '/actions_step_' + str(0) + '.npy'
states_file_name_0 = demos_expe_dir + '/states_step_' + str(0) + '.npy'
actions_0 = np.load(actions_file_name_0)
states_0 = np.load(states_file_name_0)
state_dim = states_0.shape[2]
action_dim = actions_0.shape[2]
batch_size = states_0.shape[1]
rollout_size = states_0.shape[0]
action_space = Box(low=-1, high=1, shape=(action_dim,))


class Shaping(nn.Module):
  """Module predicting the k-th shaping associated with a transition s,s'."""

  def __init__(self, input_size, hidden_size, k_max):
    super(Shaping, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, k_max)

  def forward(self, state, state_, k):
    hidden_f = torch.tanh(self.fc1(state))
    hidden_f_ = torch.tanh(self.fc1(state_))
    f_k = self.fc2(hidden_f)[:, k]
    f_k_ = self.fc2(hidden_f_)[:, k]
    return f_k, f_k_


class KL(nn.Module):
  """Module predicting KL div between two consecutive learner policies."""

  def __init__(self, input_size, hidden_size, k_max):
    super(KL, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, k_max)

  def forward(self, state, k):
    hidden_kl = torch.tanh(self.fc1(state)) # hyperbolic tangent function
    kl_k = self.fc2(hidden_kl)[:, k]
    return kl_k


def main():
  # learn policies for KL
  # ---------------------
  policies_sd = []
  policy = Policy([state_dim], action_space)
  optimizer_policy = torch.optim.Adam(policy.parameters(), lr=lr_policies)
  # observer_steps = range(1,4)
  # regression
  for k in observer_steps:

    for epoch in range(n_epoch_policies):

      actions_file_name = demos_expe_dir + '/actions_step_' + str(k) + '.npy'
      states_file_name = demos_expe_dir + '/states_step_' + str(k) + '.npy'
      actions = np.load(actions_file_name)
      states = np.load(states_file_name)

      # sample indexes
      indexes = np.random.choice(
          range(rollout_size), n_demos_per_update, replace=False)

      # create batch from indexes
      obs_batch = torch.from_numpy(states[indexes, 0]).float().view(
          -1, state_dim)

      actions_batch = torch.from_numpy(actions[indexes, 0]).float().view(
          -1, action_dim)

      # prediction
      _, action_log_probs, dist_entropy = policy.evaluate_actions(
          obs_batch, None, actions_batch)

      # loss policy
      loss = -(action_log_probs.mean() + entropy_coef*dist_entropy)

      # grad step
      optimizer_policy.zero_grad()
      loss.backward()
      optimizer_policy.step()

      if epoch % 100 == 0:
        print(
            'regression policy', k, 'epoch', epoch, 'loss_policy', loss.item())

    policies_sd.append(policy.state_dict())

    print('____________start new regression for policy', k, '_____________')

  # save last inferred policy (as a starting point for observer)
  last_policy_sd = policies_sd[-1]
  torch.save(last_policy_sd, args.policies_dir+'/last_policy_' +args.env_name +'_' +args.expe+'.pth')

  policies = []
  for pi_state_dict in policies_sd:
    pi = Policy([state_dim], action_space, base={'recurrent': False})
    pi.load_state_dict(pi_state_dict)
    policies.append(pi)

  # learn KLs
  #----------
  kl_net = KL(state_dim, 128, observer_steps[-1])
  optimizer_kl = torch.optim.Adam(kl_net.parameters(), lr=lr_kl)

  for epoch in range(n_epoch_kl):

    loss_kl = 0
    for i, k in enumerate(observer_steps[:-1]):

      actions_file_name = demos_expe_dir + '/actions_step_' + str(k) + '.npy'
      states_file_name = demos_expe_dir + '/states_step_' + str(k) + '.npy'
      actions = np.load(actions_file_name)
      states = np.load(states_file_name)

      # sample indexes
      indexes = np.random.choice(
          range(rollout_size-1), n_demos_per_update, replace=False)

      # create batch from indexes
      obs_batch = torch.from_numpy(states[indexes, 0]).float().view(
          -1, state_dim)
      next_obs_batch = torch.from_numpy(states[indexes + 1, 0]).float().view(
          -1, state_dim)
      actions_batch = torch.from_numpy(actions[indexes, 0]).float().view(
          -1, action_dim)

      # reward target
      pi_1 = policies[i]
      pi_2 = policies[i+1]

      _, log_pi_2_target, _ = pi_2.evaluate_actions(
          obs_batch, None, actions_batch)
      _, log_pi_1_target, _ = pi_1.evaluate_actions(
          obs_batch, None, actions_batch)

      kl_target = log_pi_1_target - log_pi_2_target
      kl_pred = kl_net(obs_batch, k)

      # loss policy
      loss_kl += ((kl_pred - kl_target.detach())**2).mean()

    # grad step
    optimizer_kl.zero_grad()
    loss_kl.backward()
    optimizer_kl.step()

    if epoch % 10 == 0:
      print('epoch', epoch, 'loss kl', loss_kl.item())

  # init reward with last rollout:
  # ------------------------------
  reward = Policy([state_dim], action_space, base={'recurrent': False})
  optimizer_reward = torch.optim.Adam(reward.parameters(), lr=lr_reward_init)

  k = observer_steps[-1]
  for epoch in range(n_epoch_reward_init):

    actions_file_name = demos_expe_dir + '/actions_step_' + str(k) + '.npy'
    states_file_name = demos_expe_dir + '/states_step_' + str(k) + '.npy'
    actions = np.load(actions_file_name)
    states = np.load(states_file_name)

    # sample indexes
    indexes = np.random.choice(
        range(rollout_size-1), n_demos_per_update, replace=False)

    # create batch from indexes
    obs_batch = torch.from_numpy(states[indexes, 0]).float().view(-1, state_dim)
    actions_batch = torch.from_numpy(actions[indexes, 0]).float().view(
        -1, action_dim)

    # prediction
    _, action_log_probs, _ = reward.evaluate_actions(
        obs_batch, None, actions_batch)

    # loss policy
    loss_reward = - action_log_probs.mean()

    # grad step
    optimizer_reward.zero_grad()
    loss_reward.backward()
    optimizer_reward.step()

    if epoch % 10 == 0:
      print('epoch', epoch, 'loss init reward', loss_reward.item())

  # save initial reward
  #------------------
  torch.save(reward.state_dict(),
             args.rewards_dir + '/initial_reward_' + str(args.env_name) + '_' + args.expe + '.pth')

  # Optimize shaping:
  #-----------------
  shaping = Shaping(state_dim, 128, observer_steps[-1])
  optimizer_shaping = torch.optim.Adam(
      tuple(reward.parameters()) + tuple(shaping.parameters()),
      lr=lr_reward_shaping)

  alpha = 0.7

  for epoch in range(n_epoch_reward_shaping):

    loss_shaping = 0
    for i, k in enumerate(observer_steps[:-1]):

      actions_file_name = demos_expe_dir + '/actions_step_' + str(k) + '.npy'
      states_file_name = demos_expe_dir + '/states_step_' + str(k) + '.npy'
      actions = np.load(actions_file_name)
      states = np.load(states_file_name)

      # sample indexes
      indexes = np.random.choice(
          range(rollout_size - 1), n_demos_per_update, replace=False)

      # create batch from indexes
      obs_batch = torch.from_numpy(states[indexes, 0]).float().view(
          -1, state_dim)
      next_obs_batch = torch.from_numpy(states[indexes + 1, 0]).float().view(
          -1, state_dim)
      actions_batch = torch.from_numpy(actions[indexes, 0]).float().view(
          -1, action_dim)

      # reward target
      pi_2 = policies[i + 1]
      kl = kl_net(next_obs_batch, k)

      _, log_pi_2, _ = pi_2.evaluate_actions(obs_batch, None, actions_batch)

      reward_target = log_pi_2 + gamma * kl

      # shaping
      f, f_ = shaping(obs_batch, next_obs_batch, k)

      # prediction
      _, reward_sa, _ = reward.evaluate_actions(
          obs_batch, None, actions_batch)

      shaped_reward = reward_sa + f.view(-1, 1) - gamma * f_.view(-1, 1)

      # loss policy
      loss_shaping += ((shaped_reward - alpha*reward_target.detach())**2).mean()

    # grad step
    optimizer_shaping.zero_grad()
    loss_shaping.backward()
    optimizer_shaping.step()

    if epoch % 10 == 0:
      print('epoch', epoch, 'loss shaping', loss_shaping.item())

  # save final reward
  #------------------
  torch.save(
      reward.state_dict(), args.rewards_dir + '/reward_lfl_' +args.env_name + '_' + args.expe + '.pth')


if __name__ == '__main__':
  main()
