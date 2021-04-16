"""
This code is an adaptation of the code of LfL for MuJoCo taken from https://github.com/alexis-jacq/Learning_from_a_Learner

Implements a rollout-based replay buffer for Reinforcement learning.


"""

import torch
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import SubsetRandomSampler


def _flatten_helper(t, n, tensor):
  return tensor.view(t * n, *tensor.size()[2:])


class RolloutStorage(object):
  """Rollout-based replay buffer for Reinforcement learning."""

  def __init__(self, num_steps, num_processes, obs_shape, action_space, len_rbf=None):
    self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
    self.rewards = torch.zeros(num_steps, num_processes, 1)
    self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
    self.returns = torch.zeros(num_steps + 1, num_processes, 1)
    self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
    if len_rbf is not None:
      self.feature_rewards = torch.zeros(num_steps, num_processes, len_rbf)
    else:
      self.feature_rewards = None
    if action_space.__class__.__name__ == 'Discrete':
      action_shape = 1
    else:
      action_shape = action_space.shape[0]
    self.actions = torch.zeros(num_steps, num_processes, action_shape)
    if action_space.__class__.__name__ == 'Discrete':
      self.actions = self.actions.long()
    self.masks = torch.ones(num_steps + 1, num_processes, 1)

    self.num_steps = num_steps
    self.step = 0

  def to(self, device):
    self.obs = self.obs.to(device)
    self.rewards = self.rewards.to(device)
    self.value_preds = self.value_preds.to(device)
    self.returns = self.returns.to(device)
    self.action_log_probs = self.action_log_probs.to(device)
    self.actions = self.actions.to(device)
    self.masks = self.masks.to(device)

  def insert(self, obs, actions, action_log_probs, value_preds, rewards, masks, feat=None):
    self.obs[self.step + 1].copy_(obs)
    self.actions[self.step].copy_(actions)
    self.action_log_probs[self.step].copy_(action_log_probs)
    self.value_preds[self.step].copy_(value_preds)
    self.rewards[self.step].copy_(rewards)
    self.masks[self.step + 1].copy_(masks)
    if self.feature_rewards is not None:
      self.feature_rewards[self.step].copy_(torch.Tensor(feat))

    self.step = (self.step + 1) % self.num_steps

  def after_update(self):
    self.obs[0].copy_(self.obs[-1])
    self.masks[0].copy_(self.masks[-1])

  def compute_returns(self, next_value, gamma, tau):
    """Compute returns based on generalize advantage expectation (GAE)."""

    self.value_preds[-1] = next_value
    gae = 0
    for step in reversed(range(self.rewards.size(0))):
      delta = self.rewards[step] + gamma * self.value_preds[step + 1] *\
          self.masks[step + 1] - self.value_preds[step]
      gae = delta + gamma * tau * self.masks[step + 1] * gae
      self.returns[step] = gae + self.value_preds[step]

  def feed_forward_generator(self, advantages, num_mini_batch):
    """Samples a batch of rollouts trajectories."""

    num_steps, num_processes = self.rewards.size()[0:2]
    batch_size = num_processes * num_steps
    assert batch_size >= num_mini_batch, (
        'PPO requires the number of processes ({}) '
        '* number of steps ({}) = {} '
        'to be greater than or equal to the number of PPO mini batches ({}).'
        ''.format(num_processes,
                  num_steps,
                  num_processes * num_steps,
                  num_mini_batch))
    mini_batch_size = batch_size // num_mini_batch
    sampler = BatchSampler(
        SubsetRandomSampler(
            range(batch_size)), mini_batch_size, drop_last=False)
    for indices in sampler:
      obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
      actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
      value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
      return_batch = self.returns[:-1].view(-1, 1)[indices]
      masks_batch = self.masks[:-1].view(-1, 1)[indices]
      old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
      adv_targ = advantages.view(-1, 1)[indices]

      yield obs_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ
