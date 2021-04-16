"""Argument parser to set hyperparameters and options for PPO.
This code is an adaptation of the code of LfL for MuJoCo taken from https://github.com/alexis-jacq/Learning_from_a_Learner

Code adapted from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail.
"""

import argparse


def get_args():
  """Argument parser to set hyperparameters for PPO."""

  parser = argparse.ArgumentParser(description='RL')

  parser.add_argument('--expe', default='0',
                      help='name your expe')

  parser.add_argument('--lr', type=float, default=7.e-4,
                      help='learning rate (default: 7e-4)')

  parser.add_argument('--eps', type=float, default=1e-5,
                      help='RMSprop optimizer epsilon (default: 1e-5)')

  parser.add_argument('--alpha', type=float, default=0.99,
                      help='RMSprop optimizer apha (default: 0.99)')

  parser.add_argument('--gamma', type=float, default=0.99,
                      help='discount factor for rewards (default: 0.99)')

  parser.add_argument('--tau', type=float, default=0.95,
                      help='gae parameter (default: 0.95)')

  parser.add_argument('--entropy-coef', type=float, default=0,
                      help='entropy term coefficient (default: 0.01)')

  parser.add_argument('--value-loss-coef', type=float, default=0.5,
                      help='value loss coefficient (default: 0.5)')

  parser.add_argument('--max-grad-norm', type=float, default=0.5,
                      help='max norm of gradients (default: 0.5)')

  parser.add_argument('--seed', type=int, default=42,
                      help='random seed (default: 1)')

  parser.add_argument(
      '--num-processes', type=int, default=16,
      help='how many training CPU processes to use (default: 16)')

  parser.add_argument('--num-steps', type=int, default=2000,
                      help='number of forward steps in A2C (default: 5)')

  parser.add_argument('--ppo-epoch', type=int, default=10,
                      help='number of ppo epochs (default: 4)')

  parser.add_argument('--num-mini-batch', type=int, default=32,
                      help='number of batches for ppo (default: 32)')

  parser.add_argument('--clip-param', type=float, default=0.2,
                      help='ppo clip parameter (default: 0.2)')

  parser.add_argument('--log-interval', type=int, default=1,
                      help='log interval, one log per n updates (default: 10)')

  parser.add_argument(
      '--save-interval', type=int, default=100,
      help='save interval, one save per n updates (default: 100)')

  parser.add_argument(
      '--eval-interval', type=int, default=None,
      help='eval interval, one eval per n updates (default: None)')

  parser.add_argument(
      '--num-env-steps', type=int, default=1000000,
      help='number of environment steps to train (default: 1000000)')

  parser.add_argument('--env-name', default='Hopper-v2',
                      help='environment to train on (default: HalfCheetah-v2)')

  parser.add_argument('--log-dir', default='/tmp/gym/',
                      help='directory to save agent logs (default: /tmp/gym)')

  parser.add_argument(
      '--demos-dir', default='learner_demos',
      help='directory to save agent demos for LFL (default: demos)')

  parser.add_argument('--scores-dir', default='learner_scores',
                      help='directory to save agent scores (default: scores/)')

  parser.add_argument('--rewards-dir', default='observer_rewards',
                      help='directory to save agent rewards (default: scores/)')

  parser.add_argument(
      '--policies-dir', default='observer_initial_policies',
      help='directory to save agent initial_policies (default: scores/)')

  parser.add_argument(
      '--save-dir', default='trained_models/',
      help='directory to save agent logs (default: trained_models/)')

  parser.add_argument('--add-timestep', action='store_true', default=False,
                      help='add timestep to observations')

  parser.add_argument(
      '--use-linear-lr-decay', action='store_true', default=True,
      help='use a linear schedule on the learning rate')

  parser.add_argument(
      '--use-linear-clip-decay', action='store_true', default=False,
      help='use a linear schedule on the ppo clipping parameter')

  parser.add_argument('--port', type=int, default=8097,
                      help='port to run the server on (default: 8097)')

  args = parser.parse_args()

  return args
