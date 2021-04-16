"""reproduces discrete spi inversion results from LfL paper (sect.6.2). from the original repository"""
from __future__ import print_function
from envs.grid_patch import Grid
from lfl.mdp_utils import sample_sa_trajectory
from utils.softmax_policy import *
import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn as nn
from utils.mdp import MDP


def lfl(n_run, tmax, kmax, trajectories=None, computeTrajectories=False):
    # set hyperparameters
    gride_size = 5
    n_states = gride_size ** 2
    n_actions = 4
    mu = np.zeros(n_states)
    mu[0] = 1
    gamma = 0.96
    alpha = 0.3
    alpha_model = 0.7
    entropy_coef = 0.01
    n_epoch = 10

    # generate a deterministic gridworld:
    g = Grid(gride_size, stochastic=False)

    # we just need the reward and dynamic of the MDP:
    r_gpomdp, p_gpomdp, _ = g.make_tables_gpomdp()
    r, p = g.make_tables()


    learner_score = []
    observer_score = []
    weights = []
    trajectories_spi = []
    for run in range(n_run):
        print('run', run)
        torch.manual_seed(run)

        # init first policy
        pi = np.ones((n_states, n_actions)) / n_actions

        # sample initial trajectory:
        if trajectories is None:
            np.random.seed(run)
            trajectory = sample_sa_trajectory(p, pi, tmax)
        else:
            trajectory = trajectories[run, 0]
            print(trajectory.shape)
            trajectory = trajectory.tolist()

        # transition estimation:
        p_ = np.ones((n_states, n_actions, n_states)) * 1e-15
        count = np.ones((n_states, n_actions, n_states)) * n_states * 1e-15
        for (s, a), (s_, _) in zip(trajectory[:-1], trajectory[1:]):
            p_[int(s), int(a), int(s_)] += 1
            count[int(s), int(a), :] += 1

        p_ /= count

        demos = [trajectory]
        policies = [torch.Tensor(pi)]

        # policy iterations
        for k in range(kmax):
            if trajectories is None:
                q = np.random.rand(n_states, n_actions)
                for _ in range(100):
                    v = np.zeros(n_states)
                    for state in range(n_states):
                        for action_ in range(n_actions):
                            v[state] += pi[state, action_] * \
                                        (q[state, action_] - alpha * np.log(pi[state, action_]))

                    q *= 0
                    for state in range(n_states):
                        for action in range(n_actions):
                            q[state, action] = r[state, action]
                            for state_ in range(n_states):
                                q[state, action] += gamma * p[state, action, state_] * v[state_]

                pi = np.zeros((n_states, n_actions))
                for state in range(n_states):
                    pi[state, :] = softmax(q[state, :] / alpha)

                # sample trajectory with new policy:
                trajectory = sample_sa_trajectory(p, pi, tmax)
            else:
                trajectory = trajectories[run, k+1]
                trajectory = trajectory.tolist()


            demos.append(trajectory)
            policies.append(torch.Tensor(pi))


        if not computeTrajectories:
            # learner  score
            mdp_to_evaluate = MDP(n_states, n_actions, p_gpomdp, r_gpomdp, mu, gamma)
            j_pi_learner = mdp_to_evaluate.policy_evaluation(pi.T)[0]
            learner_score.append(j_pi_learner)

            # estimate learner policies
            torch_p = torch.from_numpy(p_).float()
            logpi_ = tuple(nn.Parameter(torch.rand(n_states, n_actions, \
                                                   requires_grad=True)) \
                           for _ in range(kmax + 1))
            optimizer_pi = torch.optim.Adam(logpi_, lr=5e-1)
            for epoch in range(n_epoch):
                loss_pi = 0
                for k, demo in enumerate(demos):
                    demo_sas = [(s, a, s_) for (s, a), (s_, _) in zip(demo[:-1], demo[1:])]
                    for s, a, s_ in demo_sas:
                        dist = Categorical(torch.exp(logpi_[k][int(s), :]))
                        log_prob_demo = torch.log(dist.probs[int(a)])
                        loss_pi -= (log_prob_demo + entropy_coef * dist.entropy())

                optimizer_pi.zero_grad()
                loss_pi.backward()
                optimizer_pi.step()

            # create target reward functions:
            targets = []
            for k, demo in enumerate(demos[:-1]):
                dist_2 = torch.exp(logpi_[k + 1]) \
                         / torch.exp(logpi_[k + 1]).sum(1, keepdim=True)
                dist_1 = torch.exp(logpi_[k]) / torch.exp(logpi_[k]).sum(1, keepdim=True)
                kl = torch.log(dist_2) - torch.log(dist_1)
                r_shape = torch.zeros(n_states, n_actions)
                for state in range(n_states):
                    for action in range(n_actions):
                        r_shape[state, action] = alpha_model \
                                                 * torch.log(dist_2[state, action])
                        for state_ in range(n_states):
                            for action_ in range(n_actions):
                                r_shape[state, action] -= alpha_model * gamma \
                                                          * (kl[state_, action_]) * torch_p[state, action, state_] \
                                                          * dist_1[state_, action_]

                targets.append(r_shape)

            # recover state-action reward and shaping
            r_ = nn.Parameter(torch.zeros(n_states, n_actions, requires_grad=True))
            r_sh = (r_,) + tuple(nn.Parameter(torch.zeros(n_states, requires_grad=True)) \
                                 for _ in range(kmax))
            optimizer = torch.optim.Adam(r_sh, lr=1)
            for epoch in range(200):
                loss = 0
                for k, target in enumerate(targets):
                    loss += \
                        ((r_sh[0] + r_sh[k + 1].repeat(n_actions, 1).t() - gamma * \
                          torch.sum(torch_p * r_sh[k + 1].repeat(n_states, n_actions, 1), 2) \
                          - target.detach()) ** 2).sum()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            r_ = r_.detach().numpy()

            # solve with r_:
            mdp = MDP(n_states, n_actions, p_gpomdp, r_.T, mu, gamma)
            pi_observer = mdp.get_best_policy()

            # observer score with true reward:
            mdp_to_evaluate = MDP(n_states, n_actions, p_gpomdp, r_gpomdp, mu, gamma)
            j_pi_observer = mdp_to_evaluate.policy_evaluation(pi_observer)[0]

            observer_score.append(j_pi_observer)
            weights.append(r_)
        else:
            trajectories_spi.append(demos)
    np.save('../results/comparison_learn/lfl_SPI-v3/lfl_svi_'+''+str(kmax+1), observer_score)
    np.save('../results/comparison_learn/lfl_SPI-v3/weights_svi_'+''+str(kmax+1), weights)



if __name__ == '__main__':
    tmax = [50*20]
    kmax = [2, 4, 6, 8, 10, 12]
    n_runs = 20

    trajectories = []
    for n in range(n_runs):
        t = np.load('../agent_trajectories/SPI-v3/traj_SPI-v2_'+str(n)+'.npy')
        trajectories.append(t)
    trajectories = np.array(trajectories)
    print(trajectories.shape)

    for t in tmax:
        for k in kmax:
            traj = trajectories[:, :k, :, :]
            traj = traj.reshape(n_runs, k, 1000, 2)
            lfl(n_runs, tmax=t, kmax=k-1, trajectories=traj)
