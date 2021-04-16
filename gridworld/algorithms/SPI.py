"""
Soft-Policy improvement implementation
"""
from utils.softmax_policy import *

class SPI():

    def __init__(self, pi, n_states, n_actions, gamma, p, r):
        """
        pi = S*A
        p = A*S*S
        r = A*S
        """
        self.pi = pi
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = 0.3
        self.p = p
        self.r = r
        self.n_iterations = 100

    def update(self):
        """Soft Policy iteration"""
        q = np.random.rand(self.n_states, self.n_actions)
        for _ in range(100):
            v = np.zeros(self.n_states)
            for state in range(self.n_states):
                for action_ in range(self.n_actions):
                    v[state] += self.pi[state, action_] * \
                                (q[state, action_] - self.alpha * np.log(self.pi[state, action_]))

            q *= 0
            for state in range(self.n_states):
                for action in range(self.n_actions):
                    q[state, action] = self.r[state, action]
                    for state_ in range(self.n_states):
                        q[state, action] += self.gamma * self.p[state, action, state_] * v[state_]

        pi = np.zeros((self.n_states, self.n_actions))
        for state in range(self.n_states):
            pi[state, :] = softmax(q[state, :] / self.alpha)

        return pi
