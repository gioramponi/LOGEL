"""Parallelizes environements instead of agents.
This code is an adaptation of the code of LfL for MuJoCo taken from https://github.com/alexis-jacq/Learning_from_a_Learner

Function make_vec_envs wrapes a list of environemnts
into one larger dimensional environment that returns
a list of rewards and a concatenated observation at each step.

Taken from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
"""

import os
from baselines import bench
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize as VecNormalize_
import gym
from gym.spaces.box import Box
import numpy as np
import torch


def make_env(env_id, seed, rank, log_dir, add_timestep, allow_early_resets):
  """Returns a function that..."""
  def _thunk():
    """Creates an env and manualy sets its seed, log directory and timestep."""
    # env_id = 'Reacher'
    env = gym.make(env_id)
    env.seed(seed + rank)

    obs_shape = env.observation_space.shape

    if add_timestep and len(
        obs_shape) == 1 and str(env).find('TimeLimit') > -1:
      env = AddTimestep(env)

    if log_dir is not None:
      env = bench.Monitor(env, os.path.join(log_dir, str(rank)),
                          allow_early_resets=allow_early_resets)

    return env

  return _thunk


def make_vec_envs(env_name, seed, num_processes, gamma, log_dir, add_timestep,
                  device, allow_early_resets, num_frame_stack=None):
  """Clones and concatenates a list of occurences of an environment."""

  envs = [make_env(env_name, seed, i, log_dir, add_timestep, allow_early_resets)
          for i in range(num_processes)]

  if len(envs) > 1:
    envs = SubprocVecEnv(envs)
  else:
    # HACK for python2
    # use DummyVecEnv directly with python3
    envs = Python2DummyVecEnv(envs)

  if len(envs.observation_space.shape) == 1:
    if gamma is None:
      envs = VecNormalize(envs, ret=False)
    else:
      envs = VecNormalize(envs, gamma=gamma)
  envs = VecPyTorch(envs, device)

  return envs


class AddTimestep(gym.ObservationWrapper):
  """Add a timestep attribute to a gym environment."""

  def __init__(self, env=None):
    super(AddTimestep, self).__init__(env)
    self.observation_space = Box(self.observation_space.low[0],
                                 self.observation_space.high[0],
                                 [self.observation_space.shape[0] + 1],
                                 dtype=self.observation_space.dtype)

  def observation(self, observation):
    return np.concatenate((observation, [self.env._elapsed_steps]))


class VecPyTorch(VecEnvWrapper):
  """Wraps an env so actions, observations and rewards are Pytorch tensors."""

  def __init__(self, venv, device):
    """Return only every 'skip'-th frame."""
    super(VecPyTorch, self).__init__(venv)
    self.device = device

  def reset(self):
    obs = self.venv.reset()
    obs = torch.from_numpy(obs).float().to(self.device)
    return obs

  def step_async(self, actions):
    actions = actions.squeeze(1).cpu().numpy()
    self.venv.step_async(actions)

  def step_wait(self):
    # print('venv',self.venv)
    obs, reward, done, info = self.venv.step_wait()
    obs = torch.from_numpy(obs).float().to(self.device)
    reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
    return obs, reward, done, info


class VecNormalize(VecNormalize_):
  """Normalize observations of a gym environment."""

  def __init__(self, *args, **kwargs):
    print(*args)
    super(VecNormalize, self).__init__(*args, **kwargs)
    print('aaaa')
    self.training = True

  def _obfilt(self, obs):
    if self.ob_rms:
      if self.training:
        self.ob_rms.update(obs)
      obs = np.clip((obs - self.ob_rms.mean) /
                    np.sqrt(self.ob_rms.var + self.epsilon),
                    -self.clipob, self.clipob)
      # print(obs)
      return obs
    else:
      return obs

  def train(self):
    self.training = True

  def eval(self):
    self.training = False


class Python2DummyVecEnv(DummyVecEnv):
  """Hack for using dummyvecenv with python2."""

  def step_wait(self):
    for e in range(self.num_envs):
      obs, self.buf_rews[e], self.buf_dones[e], self.buf_infos[e]\
          = self.envs[e].step(self.actions[e])
      if self.buf_dones[e]:
        obs = self.envs[e].reset()
      self._save_obs(e, obs)
    return (self._obs_from_buf(), np.copy(self.buf_rews),
            np.copy(self.buf_dones), self.buf_infos[:])
