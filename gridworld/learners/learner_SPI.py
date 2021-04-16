"""
Implementation of a Learner using Soft Policy Iteration as learning algorithm.
It produce a batch of trajectories, the parameters of the policy followed to generate each trajectories, the reward
features at each learning step and the discounted return at each learning step.
"""
from utils.softmax_policy import *
from algorithms.SPI import SPI
from envs.grid_patch import Grid as GridWorld
import argparse


def learn_trajectories(env, n_batch, traj_len, n_learner_steps, gamma=0.96, learning_rate=None):
    """
    pi = S*A
    """
    n_actions = 4
    rew_param = len(env.reward_weights)
    param_k = np.ones((env.size**2, n_actions)) / n_actions
    trajectories = np.zeros((n_learner_steps, n_batch*traj_len, 2))
    feat_rewards = np.zeros((n_learner_steps, n_batch, traj_len, rew_param))
    params = np.zeros((n_learner_steps, param_k.shape[0], param_k.shape[1]))
    r, p = env.make_tables()

    discounted_returns = []
    discount_factor_timestep = np.power(gamma * np.ones(traj_len), range(traj_len))

    for step in range(n_learner_steps):
        env.reset()
        params[step] = param_k
        traj, rewards, feat = create_batch_trajectories(param_k, n_batch, traj_len, env, rew_param=rew_param)
        trajectories[step] = traj
        feat_rewards[step] = feat
        soft_policy_iteration = SPI(param_k, env.size ** 2, n_actions, gamma, p, r)
        param_k = soft_policy_iteration.update()

        discounted_return = np.mean(np.sum(discount_factor_timestep[np.newaxis, :, np.newaxis] * rewards, axis=1),axis=0)
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
    parser.add_argument('--dir-traj', default=None,
                        help='directory where to find the trajectories')

    args = parser.parse_args()
    env = GridWorld(size=args.size)
    if args.dir_traj == None:
        dir_traj = '../agent_trajectories/SPI'
    else:
        dir_traj = args.dir_traj

    np.random.seed(args.expe)
    trajectories, feat_rewards, params, discounted_return = learn_trajectories(env, args.batch_size, args.len_traj,
                                                            args.n_learner_steps, args.gamma,
                                                            args.learning_rate)
    np.save(dir_traj + '/traj_SPI_' + str(args.expe) + '.npy', trajectories)
    np.save(dir_traj + '/feat_SPI_' + str(args.expe) + '.npy', feat_rewards)
    np.save(dir_traj + '/params_SPI_' + str(args.expe) + '.npy', params)
    np.save(dir_traj + '/disc_ret_SPI_' + str(args.expe) + '.npy', discounted_return)
