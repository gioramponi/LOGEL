"""utils for entropy-regularized discrete MDPs."""

from __future__ import print_function
import numpy as np


def softmax(x, tau=1.):
    e = np.exp(x * tau)
    z = -np.log(sum(e))
    return np.exp(x * tau + z)


def score_policy(pi, r, p, alpha, gamma):
    """Returns expected score J(pi) = v_pi(start) using soft policy evaluation."""
    n_states, n_actions, _ = p.shape
    q_pi = np.random.rand(n_states, n_actions)
    v_pi = np.zeros(n_states)
    for _ in range(1000):
        v_pi = np.zeros(n_states)
        for state in range(n_states):
            for action_ in range(n_actions):
                v_pi[state] += pi[state, action_] * \
                               (q_pi[state, action_] - alpha * np.log(pi[state, action_]))

        q_pi *= 0
        for state in range(n_states):
            for action in range(n_actions):
                q_pi[state, action] = r[state, action]
                for state_ in range(n_states):
                    q_pi[state, action] += gamma * p[state, action, state_] * v_pi[state_]

    j_pi = v_pi[0]
    return j_pi


def solve_entropy_regularized_mdp(r, p, alpha, gamma):
    """Returns optimal (soft) policy pi* and score J(pi*)."""
    n_states, n_actions, _ = p.shape
    q = np.zeros((n_states, n_actions))
    v = np.log(np.sum(np.exp(q), 1))
    # <<<<<<< HEAD
    print("r, p: ", r.shape, p.shape)
    # =======
    #
    # >>>>>>> aed0552fe0dea9129b017edf7ec4b9d4c4dcf9f2
    for _ in range(1000):
        q = r + gamma * np.sum(p * np.tile(v, (n_states, n_actions, 1)), 2)
        v = alpha * np.log(np.sum(np.exp(q / alpha), 1))

    pi_star = np.zeros((n_states, n_actions))
    for state in range(n_states):
        pi_star[state, :] = softmax(q[state, :] / alpha)

    j_pi_star = v[0]
    return pi_star, j_pi_star


def sample_sa_trajectory(p, pi, length):
    """Returns a trajectory sampled from the learner's policy pi."""
    n_states, n_actions, _ = p.shape
    trajectory = []
    state = 0
    action = np.random.choice(range(n_actions), p=pi[state, :])
    for _ in range(length):
        new_state = np.random.choice(range(n_states), p=p[state, action, :])
        new_action = np.random.choice(range(n_actions), p=pi[new_state, :])
        trajectory.append((state, action))
        state = new_state
        action = new_action
    return trajectory


def sample_sar_trajectory(p, pi, r, length):
    """Returns a trajectory sampled from the learner's policy pi."""
    n_states, n_actions, _ = p.shape
    trajectory = []
    state = 0
    action = np.random.choice(range(n_actions), p=pi[state, :])
    for _ in range(length):
        new_state = np.random.choice(range(n_states), p=p[state, action, :])
        new_action = np.random.choice(range(n_actions), p=pi[new_state, :])
        trajectory.append((state, action, r[state, action]))
        state = new_state
        action = new_action
    return trajectory