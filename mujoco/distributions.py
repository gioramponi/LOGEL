"""Modify standard PyTorch distributions so they are compatible with this code.
This code is an adaptation of the code of LfL for MuJoCo taken from https://github.com/alexis-jacq/Learning_from_a_Learner

Code adapted from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail.
"""

import torch
import torch.nn as nn
from utils import AddBias
from utils import init

# Normal
FixedNormal = torch.distributions.Normal

log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(
    self, actions).sum(-1, keepdim=True)


entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: entropy(self).sum(-1)

FixedNormal.mode = lambda self: self.mean


class DiagGaussian(nn.Module):
  """Implements a diagonal multivariate gaussian distribution."""

  def __init__(self, num_inputs, num_outputs):
    super(DiagGaussian, self).__init__()

    init_ = lambda m: init(
        m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

    self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
    self.logstd = AddBias(torch.zeros(num_outputs))

  def forward(self, x):
    action_mean = self.fc_mean(x)
    zeros = torch.zeros(action_mean.size())
    if x.is_cuda:
      zeros = zeros.cuda()

    action_logstd = self.logstd(zeros)
    return FixedNormal(action_mean, action_logstd.exp())
