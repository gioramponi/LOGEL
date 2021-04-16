"""Reconstructs a reward function from learning trajectories.

Here is the implementation of the algorithm LOGEL
"""

import glob
import os
from arguments import get_args
from gym.spaces.box import Box
from model import Policy
import numpy as np
import torch
import copy
from scipy.optimize import minimize
import torch.nn as nn


args = get_args()

args.demos_dir = args.demos_dir
args.save_dir = args.save_dir
args.scores_dir = args.scores_dir

try:
  os.makedirs(args.rewards_dir)
except OSError:
  files = glob.glob(os.path.join(args.rewards_dir, '*'+args.expe+'.pth'))
  for file in files:
    os.remove(file)
#
try:
  os.makedirs(args.policies_dir)
except OSError:
  files = glob.glob(os.path.join(args.policies_dir, '*'+args.expe+'.pth'))
  for file in files:
    os.remove(file)


entropy_coef = 0
gamma = 0.99
n_demos_per_update = 200


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
features_file_name_0 = demos_expe_dir + '/rew_feat_step_' + str(0) + '.npy'
actions_0 = np.load(actions_file_name_0)
states_0 = np.load(states_file_name_0)
features_0 = np.load(features_file_name_0)
state_dim = states_0.shape[2]
action_dim = actions_0.shape[2]
feature_dim = features_0.shape[2]
batch_size = states_0.shape[1]
rollout_size = states_0.shape[0]
action_space = Box(low=-1, high=1, shape=(action_dim,))


num_feat = 0
if args.env_name == 'Reacher-v2':
  num_feat = 26
  observer_steps = range(10, 20)
if args.env_name == 'Hopper-v2':
  num_feat = 3
  observer_steps = range(10, 20)


def learning_rates(param, grad, weight):
  param = param.detach().numpy()
  grad = grad.detach().numpy()
  param = param.flatten()
  m = np.dot(grad, weight.T)
  v1 = np.dot(m.T, m)
  return np.dot(m.T, param) / v1


def weights_norm(param, grad, lrs=None):
  sum_1 = 0
  sum_2 = 0
  for i in range(len(grad)-1):
    g = grad[i].detach().numpy()
    delta = param[i].detach().numpy()
    delta = (param[i+1].detach().numpy() - delta)
    delta = delta.reshape((-1,1))
    sum_1 += lrs[i]*np.dot(g.T, delta)
    sum_2 += lrs[i]**2*np.dot(g.T, g)
  try:
    return np.dot(sum_1.T, np.linalg.inv(sum_2))
  except:
    print('yes')
    return np.dot(sum_1.T, np.linalg.inv(sum_2+0.1*np.identity(sum_2.shape[0])))




def main():

  # ---------------------
  policies_sd = []


  policy = Policy([state_dim], action_space)
  optimizer_policy = torch.optim.Adam(policy.parameters(), lr=lr_policies)
  # regression

  ## BEHAVIORAL CLONINC
  for k in observer_steps:

    for epoch in range(n_epoch_policies):

      actions_file_name = demos_expe_dir + '/actions_step2_' + str(k) + '.npy'
      states_file_name = demos_expe_dir + '/states_step2_' + str(k) + '.npy'
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
    policies_sd.append(copy.deepcopy(policy))

    print('____________start new regression for policy', k, '_____________')

  ''' GET GRADIENTS J '''
  policy = policies_sd[0]
  p__ = [p_ for p_ in list(policy.base.actor.parameters())]
  num_parameters = len(torch.cat([w.flatten() for w in p__]))
  all_gradients = torch.zeros((len(policies_sd),num_parameters, feature_dim))
  all_thetas = torch.zeros((len(policies_sd), num_parameters))
  if args.env_name == 'Swimmer-v2':
    all_gradients = torch.zeros((len(policies_sd), num_parameters, feature_dim))
    all_thetas = torch.zeros((len(policies_sd), num_parameters))
  # create batch from indexes
  indexes = np.arange(2000)+1
  for p in range(len(observer_steps)):#len(1)):#policies_sd)):
    policy = policies_sd[p]
    actions_file_name = demos_expe_dir + '/actions_step2_' + str(p+observer_steps[0]) + '.npy'
    states_file_name = demos_expe_dir + '/states_step2_' + str(p+observer_steps[0]) + '.npy'
    features_file_name = demos_expe_dir + '/rew_feat_step2_' + str(p+observer_steps[0]) + '.npy'
    actions = np.load(actions_file_name)
    states = np.load(states_file_name)
    features = np.load(features_file_name)

    obs_batch = torch.from_numpy(states[indexes, 0]).float().reshape((-1,50,state_dim))

    actions_batch = torch.from_numpy(actions[indexes, 0]).float().reshape((-1,50,action_dim))

    features_batch = torch.from_numpy(features[indexes, 0]).float().reshape((-1,50,num_feat))


    discounted_return = .99 ** np.arange(50)
    # prediction

    features_batch = features_batch * discounted_return[None, :, None]

    weights = [p_ for p_ in list(policy.base.actor.parameters())]

    all_thetas[p] = torch.cat([w.flatten() for w in weights])

    gradients_log_policy = torch.zeros((obs_batch.shape[0], obs_batch.shape[1], 4928))
    if args.env_name == 'Swimmer-v2' or args.env_name == 'HalfCheetah-v2':
      gradients_log_policy = torch.zeros((obs_batch.shape[0], obs_batch.shape[1], 4736))

    for n in range(obs_batch.shape[0]):
      for t in range(obs_batch.shape[1]):
        policy_ = copy.deepcopy(policy)
        value, action_log_prob, dist_entropy = policy_.evaluate_actions(
          obs_batch[n,t].reshape((1,-1)), None, actions_batch[n,t].reshape((1,-1)))
        loss = action_log_prob
        loss.backward()
        weights = [p_.grad for p_ in list(policy_.base.actor.parameters())]
        w2 = torch.cat([w.flatten() for w in weights])
        gradients_log_policy[n,t] = w2

    gradient_est_timestep = torch.cumsum(gradients_log_policy, dim=1)  # (N,T,K)
    gradient_est_timestep2 = torch.cumsum(gradients_log_policy, dim=1) ** 2  # (N,T,K)
    baseline_den = torch.mean(gradient_est_timestep2, dim=0)  # (T,K)
    baseline_num = torch.mean(
          (gradient_est_timestep2)[:, :, :, np.newaxis] * features_batch[:, :, np.newaxis, :],
          dim=0)  # (T,K,L)
    baseline = baseline_num / (baseline_den[:, :, np.newaxis] + 0.0001)  # (T,K,L)

    gradient = torch.mean(torch.sum(gradient_est_timestep[:, :, :, np.newaxis] * (features_batch[:, :, np.newaxis, :] -
                                                                        baseline[np.newaxis, :]),
                          dim=1), dim=0)  # (N,T,K,L aggiunta) * (N,T,K aggiunta,L) = (N,K,L) somma rispetto a T

    all_gradients[p] = gradient

    ''' EVALUATE WEIGHTS '''
  lrs = np.random.random(len(all_gradients))*1
  weights = weights_norm(all_thetas, all_gradients, lrs)
  weights_old = np.zeros(weights.shape)
  max_steps = 300
  step = 0
  print(weights)
  while step < max_steps or np.linalg.norm(weights-weights_old)<1.e-4:
    step += 1
    weights_old = weights
    ''' EVALUATE alpha '''
    for g in range(1,len(all_gradients)):
      lrs[g-1] = learning_rates(all_thetas[g] - all_thetas[g-1], all_gradients[g-1], weights)
      if lrs[g-1] < 0:
        lrs[g-1] = 0
    weights = weights_norm(all_thetas, all_gradients, lrs)
    print ('ITERATION ', p)

  np.save(args.rewards_dir + '/reward_logel_' + args.env_name +'_' + args.expe + '.pth', weights)



if __name__ == '__main__':
  main()
