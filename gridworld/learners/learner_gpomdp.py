"""
Implementation of the learner using GPOMDP algorithm.
It produce a batch of trajectories, the parameters of the policy followed to generate each trajectories, the reward
features at each learning step and the discounted return at each learning step.
"""

from utils.boltzmann_policy import *
from algorithms.gpomdp import GPOMDP
from envs.grid_patch import Grid as GridWorld
import argparse
import numpy as np
from gradient_descent import Adam


def learn_trajectories(env, n_batch, traj_len, n_learner_steps, gamma = 0.96, learning_rate = None):
    rew_param = len(env.reward_weights)
    param_k = np.zeros((env.size**2*3,1))
    trajectories = np.zeros((n_learner_steps,n_batch,traj_len, 2))
    feat_rewards = np.zeros((n_learner_steps, n_batch, traj_len, rew_param))
    params = np.zeros((n_learner_steps, param_k.shape[0], 1))
    optimize = Adam(learning_rate=.01, ascent=True)
    optimize.initialize(param_k)

    discounted_returns = []
    discount_factor_timestep = np.power(gamma * np.ones(traj_len),range(traj_len))
    for step in range(n_learner_steps):
        env.reset()
        params[step] = param_k
        traj, rewards, feat = create_batch_trajectories(param_k, n_batch, traj_len, env, rew_param=rew_param)
        trajectories[step] = traj
        feat_rewards[step] = feat
        gradients = compute_gradients(n_batch, traj_len, traj, env, param_k)
        gpomdp = GPOMDP(gradient=np.array(gradients), gamma=gamma, rewards=rewards)
        g = gpomdp.eval_gpomdp()
        if learning_rate is None:
            alpha = .1
        elif learning_rate == 'norm':
            alpha = 1/np.linalg.norm(g)
        elif learning_rate == 'lin':
            alpha = 1/(1+step)
        elif learning_rate == 'square':
            alpha = 1/(1+step)**2
        param_k = param_k + alpha * g

        discounted_return = np.mean(np.sum(discount_factor_timestep[np.newaxis, :, np.newaxis] * rewards, axis=1),
                                    axis=0)
        discounted_returns.append(float(discounted_return))

    discounted_returns = np.array(discounted_returns)

    return trajectories, feat_rewards, params, discounted_returns


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL')

    parser.add_argument('--expe', default='2',
                        help='name your expe')
    parser.add_argument('--batch-size', default=50,
                        help='number of trajectories')
    parser.add_argument('--size', default=5,
                        help='size of the gridworld')
    parser.add_argument('--len-traj', default=20,
                        help='length of the trajectories')
    parser.add_argument('--n-learner-steps', default=30,
                        help='learning steps')
    parser.add_argument('--gamma', default=0.96,
                        help='discount factor')
    parser.add_argument('--max-steps', default=50,
                        help='maximum number of iteration to retrieve the reward weights')
    parser.add_argument('--epsilon', default=1.e-2,
                        help='parameter to stop the weight estimation')
    parser.add_argument('--learning_rate', default=None,
                        help='type of learning rate to use (norm, lin, square)')
    parser.add_argument('--n-runs', default=10,
                        help='number of runs')
    parser.add_argument('--dir-traj', default=None,
                        help='directory where to find the trajectories')

    args = parser.parse_args()
    env = GridWorld(size=args.size)
    if args.dir_traj == None:
        dir_traj = '../agent_trajectories/GPOMDP'
    else:
        dir_traj = args.dir_traj
    trajectories, feat_rewards, params, discounted_return = learn_trajectories(env, args.batch_size, args.len_traj,
                                                           args.n_learner_steps, args.gamma,
                                                           args.learning_rate)
    np.save(dir_traj + '/traj_GPOMDP_' +str(args.expe) + '.npy', trajectories)
    np.save(dir_traj + '/feat_GPOMDP_' +str(args.expe) + '.npy', feat_rewards)
    np.save(dir_traj + '/params_GPOMDP_' +str(args.expe) + '.npy', params)
    np.save(dir_traj + '/disc_ret_GPOMDP_' +str(args.expe) + '.npy', discounted_return)
