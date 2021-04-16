import numpy as np


class MDP(object):
    """
    A class representing a Markov decision process (MDP).

    Parameters
    ----------
        - n_states: the number of states (S)
        - n_actions: the number of actions (A)
        - transitions: transition probability matrix P(S_{t+1}=s'|S_t=s,A_t=a). Shape: (AxSxS)
        - rewards: the reward function R(s,a) or R(s,a,s'). Shape: (A,S) or (A,S,S)
        - init_state: the initial state probabilities P(S_0=s). Shape: (S,)
        - gamma: discount factor in [0,1]
    """
    # def __init__(self, n_states, n_actions, transitions, rewards, init_state, gamma, variance_reward,
    #             reward_prob_dict=None, reward_prob_f=None):

    def __init__(self, n_states, n_actions, transitions, rewards, init_state, gamma,
                 reward_prob_dict=None, reward_prob_f=None): # delete variance_reward, not used

        assert n_states > 0, "The number of states must be positive"
        self.S = n_states
        assert n_actions > 0, "The number of actions must be positive"
        self.A = n_actions

        assert 0 <= gamma < 1, "Gamma must be in [0,1)"
        self.gamma = gamma

        assert transitions.shape == (n_actions, n_states, n_states), "Wrong shape for P"
        self.P = [transitions[a] for a in range(self.A)]
        if rewards.shape == (n_actions, n_states, n_states):
            # Compute the expected one-step rewards
            self.R = np.sum(self.P * rewards, axis=2)
        elif rewards.shape == (n_actions, n_states):
            self.R = rewards
        else:
            raise TypeError("Wrong shape for R")

        # assert variance_reward.shape == (n_actions, n_states), "The shape of variance"
        # self.variance_reward = variance_reward

        assert init_state.shape == (n_states,), "Wrong shape for P0"
        self.P0 = init_state

        self.V = None
        self.pi = None
        self.reward_prob_dict = reward_prob_dict  # dictionary where keys are possible rewards and values are np.arrays (A, S)
        # containing probabilities
        self.reward_prob_f = reward_prob_f

    def compute_reward_probability(self, state, action, reward):
        """
        Compute the probability of observing a certain reward in a given state
        """
        if self.reward_prob_dict is not None:
            return self.reward_prob_dict[reward][action, state]
        if self.reward_prob_f is not None:
            return self.reward_prob_f(state, action, reward)
        raise RuntimeError("The lambda function has not been initialized")

    def get_Q_function(self, max_iter=1, tol=1e-3, verbose=False, compute=False):
        """
        :return Q function: Shape(A,S)
        """
        if self.V is None or compute:
            self.pi, self.V = self.value_iteration(max_iter=max_iter, tol=tol, verbose=verbose)
        Q = self.R + np.dot(self.P, self.V)
        return Q

    def get_best_policy(self, max_iter=1000, tol=1e-3, verbose=False, compute=False):
        """
        Return the best policy of the MDP, in the shape of: (S,).
        Best action for each state
        """
        # policy = np.zeros(shape=self.S)
        # Q = self.get_Q_function(max_iter=max_iter, verbose=verbose, tol=tol, compute=compute)
        # for state in self.S:
        #    policy[state] = np.argmax(Q[:, state])
        # return policy
        if self.pi is None or compute:
            self.pi, self.V = self.value_iteration(max_iter=max_iter, tol=tol, verbose=verbose)
        return self.pi

    def get_value_function(self, max_iter=1000, tol=1e-3, verbose=False, compute=False):
        """
        :return value function of the MPD
        """
        if self.V is None or compute:
            self.pi, self.V = self.value_iteration(max_iter=max_iter, tol=tol, verbose=verbose)
        return self.V

    def compute_variance_environment(self, value_function):
        """
        Compute the variance of a value function in each state-action pair
        according to the environment dynamics.

        :param value_function value function on which the variance will be computed
        """
        variance = np.zeros(shape=(self.A, self.S))
        for state in range(0, self.S):
            for action in range(0, self.A):
                variance[action, state] = self.compute_variance_value_function(value_function=value_function,
                                                                               state=state,
                                                                               action=action)

        return variance

    def compute_variance_value_function_vector(self, V_vector: np.array, state: int, action: int):
        """
        Compute the variance for a vector of value functions

        :param V_vector vector of value functions of which the variance will be computed
        :param state state in which the action is taken
        :param action action taken in the given state
        :return np.array containing the variance of each value function
        """
        variance_vector = np.zeros(shape=V_vector.shape[0])
        for i, value_function in enumerate(V_vector):
            variance_vector[i] = self.compute_variance_value_function(value_function=value_function, state=state,
                                                                      action=action)
        return variance_vector

    def compute_variance_value_function(self, value_function: np.array, state: int, action: int):
        """
        Compute the variance of the value function according to the following formula:
            dot_product(p(s,a),  V)

        :param value_function: value function of which the variance will be computed
        :param state: state in which the action is taken
        :param action: action taking in the given state
        :return dot_product(p(s,a), V)
        """
        if value_function.shape != (self.S,):
            raise AttributeError("Value function has an incorrect number of states. {} expected, {} found".
                                 format(value_function.shape[0], self.S))
        mean = np.dot(self.P[action][state], value_function)
        moment_second_order = np.dot(self.P[action][state], np.square(value_function))
        return moment_second_order - (mean ** 2)

    def compute_transition_per_value_function_environment(self, value_function):
        """
        P*V environment-wise

        :param value_function value function on which the the product will be computed against the transition
        matrix
        :return: P*V in each state
        """
        result = np.zeros(shape=(self.A, self.S))
        for state in range(0, self.S):
            for action in range(0, self.A):
                result[action, state] = self.compute_transition_per_value_function_(value_function=value_function,
                                                                                    state=state,
                                                                                    action=action)

        return result

    def compute_transition_per_value_function_vector(self, V_vector: np.array, state: int, action: int):
        """
        Compute the product of the transition with a value function vector for a vector of value functions in a
        given state-action pair.

        :param V_vector vector of value functions of which the product will be computed
        :param state state in which the action is taken
        :param action action taken in the given state
        :return np.array containing the product against each value function
        """
        variance_vector = np.zeros(shape=V_vector.shape)
        for i, value_function in enumerate(V_vector):
            variance_vector[i] = self.compute_transition_per_value_function_(value_function=value_function, state=state,
                                                                             action=action)
        return variance_vector

    def compute_transition_per_value_function_(self, value_function: np.array, state: int, action: int):
        """
        dot_product(p(s,a),  V)

        :param value_function: value function of which the product will be computed against the transition of this MDP
        :param state: state in which the action is taken
        :param action: action taking in the given state
        :return dot_product(p(s,a), V)
        """
        if value_function.shape != (self.S,):
            raise AttributeError("Value function has an incorrect number of states. {} expected, {} found".
                                 format(value_function.shape[0], self.S))
        return np.dot(self.P[action][state], value_function)

    def bellman_op(self, V):
        """
        Applies the optimal Bellman operator to a value function V.

        :param V: a value function. Shape: (S,)
        :return: the updated value function and the corresponding greedy action for each state. Shapes: (S,) and (S,)
        """

        assert V.shape == (self.S,), "V must be an {0}-dimensional vector".format(self.S)

        Q = np.empty((self.A, self.S))
        for a in range(self.A):
            Q[a] = self.R[a] + self.gamma * self.P[a].dot(V)

        return Q.argmax(axis=0), Q.max(axis=0)



    def value_iteration(self, max_iter=1, tol=1e-3, verbose=False):
        """
        Applies value iteration to this MDP.

        :param max_iter: maximum number of iterations
        :param tol: tolerance required to converge
        :param verbose: whether to print info
        :return: the optimal policy and the optimal value function.

        """

        # Initialize the value function to zero
        V = np.zeros(self.S, )

        for i in range(max_iter):

            # Apply the optimal Bellman operator to V
            pi, V_new = self.bellman_op(V)

            # Check whether the difference between the new and old values are below the given tolerance
            diff = np.max(np.abs(V - V_new))

            # if verbose:
            #     print("Iter: {0}, ||V_new - V_old||: {1}, ||V_new - V*||: {2}".format(i, diff,
            #                                                                           2 * diff * self.gamma / (
            #                                                                                   1 - self.gamma)))

            # Terminate if the change is below tolerance
            if diff <= tol:
                break

            # Set the new value function
            V = V_new

        return pi, V

    def bellman_exp_op(self, V, pi):
        """
        Applies the Bellman expectation operator for policy pi to a value function V.

        :param V: a value function. Shape: (S,)
        :param pi: a stochastic policy. Shape: (A,S)
        :return: the updated value function. Shape: (S,)
        """

        assert V.shape == (self.S,), "V must be an {0}-dimensional vector".format(self.S)

        Q = np.empty((self.A, self.S))
        for a in range(self.A):
            Q[a] = self.R[a] + self.gamma * self.P[a].dot(V)

        return np.sum(pi * Q, axis=0)

    def policy_evaluation(self, pi, max_iter=1000, tol=1e-3, verbose=False):
        """
        Evaluates policy pi on this MDP.

        :param pi: a policy. Shape (A,S) or (S,)
        :param max_iter: maximum number of iterations
        :param tol: tolerance required to converge
        :param verbose: whether to print info
        :return: the value function of policy pi. Shape: (S,)
        """

        if pi.shape == (self.S,):
            pi = self._deterministic_to_stochastic(pi)

        # Initialize the value function to zero
        V = np.zeros(self.S, )

        for i in range(max_iter):

            # Apply the optimal Bellman operator to V
            V_new = self.bellman_exp_op(V, pi)

            # Check whether the difference between the new and old values are below the given tolerance
            diff = np.max(np.abs(V - V_new))

            if verbose:
                print("Iter: {0}, ||V_new - V_old||: {1}, ||V_new - V*||: {2}".format(i, diff,
                                                                                      2 * diff * self.gamma / (
                                                                                              1 - self.gamma)))

            # Terminate if the change is below tolerance
            if diff <= tol:
                break

            # Set the new value function
            V = V_new

        return V

    def _deterministic_to_stochastic(self, pi):
        """
        Converts a deterministic policy pi to a stochastic one.

        :param pi: deterministic policy. Shape (S,)
        :return: stochastic policy. Shape (A,S)
        """

        pi_stochastic = np.zeros([self.A, self.S])
        # pi_stochastic = np.zeros(n_actions, n_states)

        # pi_stochastic = np.zeros(self.S)

        for s in range(self.S):
            pi_stochastic[pi[s], s] = 1

        return pi_stochastic


def random_mdp(n_states, n_actions, gamma=0.99):
    """
    Creates a random MDP.

    :param n_states: number of states
    :param n_actions: number of actions
    :param gamma: discount factor
    :return: and MDP with S state, A actions, and randomly generated transitions and rewards
    """

    # Create a random transition matrix
    P = np.random.rand(n_actions, n_states, n_states)
    # Make sure the probabilities are normalized
    for s in range(n_states):
        for a in range(n_actions):
            P[a, s, :] = P[a, s, :] / np.sum(P[a, s, :])

    # Create a random reward matrix
    R = np.random.rand(n_actions, n_states)

    # Create a random variance matrix
    var = np.ones(n_actions, n_states)

    # Create a random initial-state distribution
    P0 = np.random.rand(n_states)
    # Normalize
    P0 /= np.sum(P0)
    # mdpobj= MDP(n_states, n_actions, P, R, P0, gamma)

    return MDP(n_states, n_actions, P, R, P0, gamma, var)
