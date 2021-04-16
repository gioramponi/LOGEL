"""
QL algorithm's implementation for the learner learning through Q-Learning
"""
import numpy as np
from envs.grid_patch import Grid


class QL():

    def __init__(self, Q, n_states, n_actions, gamma, p, r, epsilon=0.9, n_iteration=1000):
        self.Q = Q
        self.n_states = n_states
        self.n_actions = n_actions
        self.p = p
        self.r = r
        self.gamma = gamma
        self.alpha = 0.1
        self.epsilon = epsilon
        self.n_iterations = 100

    def get_next_action(self, s):
        """Boltzmann Policy"""
        probs = np.array([self.Q[act, s] for act in range(4)]).ravel()
        probs = np.exp(probs)
        probs /= np.sum(probs)
        a = int(np.random.choice(np.arange(4), p=probs))
        return a

    def get_starting_state(self):
        state = np.random.randint(self.n_states)
        return state

    def optimal_bellman(self, Q, gamma, alpha, p, r):
        state = self.get_starting_state()
        for _ in range(self.n_iterations):
            action = self.get_next_action(state)

            previous_state = state
            state = np.argmax(p[action, state])
            reward = r[action, state]
            old_q_value = Q[action, previous_state]

            new_q_value = old_q_value + alpha * (reward + (gamma * np.max(Q[:, state])) - old_q_value)

            Q[action, previous_state] = new_q_value

        return Q.argmax(axis=0), Q

    def update(self):
        pi, Q_new = self.optimal_bellman(self.Q, self.gamma, self.alpha, self.p, self.r)
        return pi, Q_new
