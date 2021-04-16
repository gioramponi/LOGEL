"""Model used to implement both learner and observer policies.
This code is an adaptation of the code of LfL for MuJoCo taken from https://github.com/alexis-jacq/Learning_from_a_Learner

This code is an adaptation of the PPO implementation taken from
https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail.
"""

from distributions import DiagGaussian
import numpy as np
import torch.nn as nn
from utils import init


class Flatten(nn.Module):

  def forward(self, x):
    return x.view(x.size(0), -1)


class Policy(nn.Module):
  """Policy for PPO agents in mujoco environments."""

  def __init__(self, obs_shape, action_space, base=None):
    super(Policy, self).__init__()
    self.base = MLPBase(obs_shape[0])
    num_outputs = action_space.shape[0]
    self.dist = DiagGaussian(self.base.output_size, num_outputs)

  def forward(self, inputs, masks):
    raise NotImplementedError

  def act(self, inputs, masks, deterministic=False):
    value, actor_features = self.base(inputs, masks)
    dist = self.dist(actor_features)

    if deterministic:
      action = dist.mode()
    else:
      action = dist.sample()

    action_log_probs = dist.log_probs(action)

    return value, action, action_log_probs

  def get_value(self, inputs, masks):
    value, _ = self.base(inputs, masks)
    return value

  def evaluate_actions(self, inputs, masks, action):
    value, actor_features = self.base(inputs, masks)
    dist = self.dist(actor_features)

    action_log_probs = dist.log_probs(action)
    dist_entropy = dist.entropy().mean()

    return value, action_log_probs, dist_entropy


class MLPBase(nn.Module):
  """Pytorch Module base for PPO policy."""

  def __init__(self, num_inputs, hidden_size=64):
    super(MLPBase, self).__init__()

    self.output_size = hidden_size

    init_ = lambda m: init(
        m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))


    self.actor = nn.Sequential(
        init_(nn.Linear(num_inputs, hidden_size)),
        nn.Tanh(),
        init_(nn.Linear(hidden_size, hidden_size)),
        nn.Tanh()
        )

    self.critic = nn.Sequential(
        init_(nn.Linear(num_inputs, hidden_size)),
        nn.Tanh(),
        init_(nn.Linear(hidden_size, hidden_size)),
        nn.Tanh()
        )

    self.critic_linear = init_(nn.Linear(hidden_size, 1))

    self.train()

  def forward(self, inputs, masks):
    x = inputs

    hidden_critic = self.critic(x)
    hidden_actor = self.actor(x)

    return self.critic_linear(hidden_critic), hidden_actor
