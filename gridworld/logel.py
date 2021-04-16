from envs.grid_patch import Grid as GridWorld
from algorithms.gpomdp import GPOMDP
from utils.mdp import MDP
import torch
from torch import nn
from utils.boltzmann_policy import *
from torch.distributions import Categorical
import argparse
from learners.learner_gpomdp import learn_trajectories


def weights_norm(param, grad, lrs=None,  learning_step=5):
  sum_1 = 0
  sum_2 = 0
  for i in range(learning_step):
    g = grad[i]
    delta = param[i]
    delta = delta.reshape((-1,1))
    sum_1 += lrs[i]*np.dot(g.T, delta)
    sum_2 += lrs[i]**2*np.dot(g.T, g)
  try:
    return np.dot(sum_1.T, np.linalg.inv(sum_2)).flatten()
  except:
    return np.dot(sum_1.T, np.linalg.inv(sum_2+0.01*np.identity(sum_2.shape[0]))).flatten()

def learning_rates(param, grad, weight):
    """
    Function to compute the learning rates given the other parameters, in closed form.
    :param param: Policy parameters
    :param grad: Gradients
    :param weight: Reward weights
    :return: Learning rates
    """
    param = param.flatten()
    m = np.dot(grad, weight)
    v1 = np.dot(m.T, m)
    return np.dot(param, m) / v1


def run(args):
    env = GridWorld(size=args.size)
    rew_param = len(env.reward_weights)
    mu = np.zeros((env.size ** 2))
    start_int = env._coupleToInt(env.start[0], env.start[1])
    mu[start_int] = 1
    rew_matrix, trans_matrix, r_f = env.make_tables_gpomdp()
    observed_list = np.arange(args.min_step_observed, args.max_step_observed, args.steps_of_observation).tolist()

    # COMPUTE THE TRAJECTORIES USING GPOMDP IF NOT GIVEN

    if args.dir_traj is None:
        trajectories, feat_rewards, params, _ = learn_trajectories(env, args.batch_size, args.len_traj, args.n_learner_steps,
                                                                args.gamma,
                                                                args.learning_rate)
    # RETRIEVE THE GIVEN TRAJECTORIES
    else:
        trajectories_dir = args.dir_traj + '/traj_' + args.expe + '_' + str(n) + '.npy'
        params = args.dir_traj + '/params_' + args.expe + '_' + str(n) + '.npy'

        true_learner_steps = args.true_learner_steps
        trajectories = np.load(trajectories_dir)
        trajectories = trajectories[:true_learner_steps]
        params = np.load(params).reshape((true_learner_steps, -1, 1))
        params = params - np.min(params, axis=1)[:, np.newaxis, :]
        params = params[:, :75]
        trajectories = trajectories.reshape((true_learner_steps, args.batch_size, args.len_traj, 2))

        feat_rewards = np.array([compute_reward_and_feat(trajectories[step], args.batch_size, args.len_traj, env)[1]
                                 for step in range(args.n_learner_steps)])

    trajectories = trajectories[:, :args.batch_size]
    feat_rewards = feat_rewards[:, :args.batch_size]

    # BEHAVIOURAL CLONING

    n_pi_parameters = 75

    if args.bc:
        params = np.zeros((args.n_learner_steps, n_pi_parameters, 1))
        logpi_ = tuple(nn.Parameter(torch.zeros(env.size ** 2, 4 - 1, requires_grad=True)) for _ in range(1))
        for step in range(args.n_learner_steps):
            optimizer_pi = torch.optim.Adam(logpi_, lr=5e-1)
            traj = trajectories[step].reshape((-1, 2))
            for epoch in range(args.num_iterations_bc):
                loss_pi = 0
                demo_sas_0 = [(int(s), int(a)) for (s, a) in traj]
                for s, a in demo_sas_0:
                    dist = Categorical(torch.cat((torch.exp(logpi_[0][s, :]), torch.Tensor(torch.ones(1))), 0))
                    log_prob_demo = torch.log(dist.probs[a])
                    loss_pi -= (log_prob_demo + 0.01 * dist.entropy())
                optimizer_pi.zero_grad()
                loss_pi.backward()
                optimizer_pi.step()
            params[step] = logpi_[0].detach().numpy().reshape((-1, 1))


    # EVALUATE GRADIENTS AND DELTA

    jacobians = np.zeros((args.n_learner_steps - 1, params.shape[1], rew_param))
    deltas = np.zeros((args.n_learner_steps - 1, params.shape[1], 1))
    for step in range(args.n_learner_steps - 1):
        deltas[step] = params[step + 1] - params[step]
        gradients = compute_gradients(args.batch_size, args.len_traj, trajectories[step], env, params[step])
        gpomdp = GPOMDP(gradient=np.array(gradients), gamma=args.gamma, rewards=feat_rewards[step])
        jacobians[step] = gpomdp.eval_gpomdp()

    # EVALUATE LRS AND REWARDS

    lrs_init = np.ones(args.n_learner_steps - 1) * .1
    w = weights_norm(deltas, jacobians, lrs_init, learning_step=1)
    max_steps = args.max_steps
    epsilon = args.epsilon
    epsilon_lr = args.epsilon_lr
    w_init = np.zeros(w.shape)
    weights = np.zeros((len(observed_list), w.shape[0]))

    for it, lear_step in enumerate(observed_list):
        step = 0
        w = weights_norm(deltas[:lear_step], jacobians[:lear_step], lrs=lrs_init[:lear_step],
                         learning_step=lear_step)
        while np.linalg.norm(w - w_init) > epsilon and step < max_steps:
            step += 1
            lrs = []
            for s in range(lear_step):
                new_lr = learning_rates(deltas[s], jacobians[s], w)
                if new_lr < epsilon_lr:
                    new_lr = epsilon_lr
                lrs.append(new_lr)
            w_init = w
            w = weights_norm(deltas[:lear_step], jacobians[:lear_step], lrs=lrs[:lear_step],
                             learning_step=lear_step)
        weights[it] = w
    mdp_to_evaluate = MDP(env.size ** 2, 4, trans_matrix, rew_matrix, mu, args.gamma)

    # CREATION OF DIFFERENT GRIDWORLDS WITH THE DIFFERENT COMPUTED WEIGHTS

    value = np.zeros(len(observed_list))
    for it, w in enumerate(weights):
        env_j = GridWorld(size=env.size, reward_weights=w)
        rew_matrix_j, _, _ = env_j.make_tables_gpomdp()
        _, trans_matrix_j, _ = env_j.make_tables_gpomdp()
        mdp_j = MDP(env_j.size ** 2, 4, trans_matrix_j, rew_matrix_j, mu, args.gamma)
        # computation of the best policy for that gridworld
        pi = mdp_j.get_best_policy()
        # evaluation of the computed policy for the gridworld with original weights
        value[it] = mdp_to_evaluate.policy_evaluation(pi)[0]
    np.save(args.dir_output + '/value_logel_' + str(args.expe) + '_' + str(args.alg_name) + '.npy', value)
    np.save(args.dir_output + '/weights_logel_' + str(args.expe) + '_' + str(args.alg_name) + '.npy', weights)


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
    parser.add_argument('--n-learner-steps', default=16,
                        help='number of analyzed learning steps')
    parser.add_argument('--true-learner-steps', default=30,
                        help='number of learning steps taken by the learner')
    parser.add_argument('--min-step-observed', default=5,
                        help='minimum learning step to consider')
    parser.add_argument('--max-step-observed', default=20,
                        help='maximum learning step to consider')
    parser.add_argument('--steps-of-observation', default=5,
                        help='steps of learning steps to consider')
    parser.add_argument('--gamma', default=0.96,
                        help='discount factor')
    parser.add_argument('--alg_name', default='GPODMP',
                        help='name of the algorithm')
    parser.add_argument('--max-steps', default=50,
                        help='maximum number of iteration to retrieve the reward weights')

    parser.add_argument('--epsilon', default=1.e-2,
                        help='parameter to stop the weight estimation')
    parser.add_argument('--epsilon_lr', default=0.01,
                        help='constraint of the learning rate')
    parser.add_argument('--learning_rate', default=None,
                        help='type of learning rate to use (norm, lin, square, adam)')
    parser.add_argument('--num-iterations-bc', default=3,
                        help='iteration of the behavioural cloning')
    parser.add_argument('--bc', default=False,
                        help='flag to enable the behavioural cloning')
    parser.add_argument('--dir-traj', default=None,
                        help='directory where to find the trajectories')
    parser.add_argument('--dir-output', default='results',
                        help='directory where to store the results')

    args = parser.parse_args()


    run(args)
