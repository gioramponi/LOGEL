"""
Implementation of Soft Value Iteration
"""

import numpy as np


class SVI():

    def __init__(self, Q, n_states, n_actions, gamma, p, r, pi):
        self.Q = Q
        self.n_states = n_states
        self.n_actions = n_actions
        self.p = p
        self.r = r
        self.gamma = gamma
        self.pi = pi

    def bellman_op(self, Q_old, n_state, n_action, gamma, p, r):
        Q = np.empty((n_action, n_state))
        for a in range(n_action):
            Q[a] = r[a] + gamma * p[a].dot(np.log(np.sum(np.exp(Q_old), axis=0)))

        return Q.argmax(axis=0), Q

    def update(self):
        pi, V_new = self.bellman_op(self.Q, self.n_states, self.n_actions, self.gamma, self.p, self.r)
        return pi, V_new