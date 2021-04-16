"""
GPOMDP algorithm's implementation for the gpomdp learner
"""
import numpy as np

class GPOMDP():
    def __init__(self, gradient, gamma, rewards, weights=None):
        self.gradients = gradient
        self.gamma = gamma # discount factor
        self.weights = weights
        self.rewards = rewards

    """
    K = number of policy paramenters
    T = horizon
    N = batch size
    L = number of reward parameters
    
    """
    def eval_gpomdp(self):
        discount_factor_timestep = np.power(self.gamma * np.ones(self.gradients.shape[1]),
                                                range(self.gradients.shape[1]))  # (T,)
        # discunted return
        discounted_return = discount_factor_timestep[np.newaxis, :, np.newaxis] * self.rewards# (N,T,L)

        gradient_est_timestep = np.cumsum(self.gradients, axis=1) # (N,T,K)

        # baseline
        gradient_est_timestep2 = gradient_est_timestep ** 2  # (N,T,K)

        baseline_den = np.mean(gradient_est_timestep2, axis=0) + 1.e-3  # (T,K)
        baseline_num = np.mean(
            (gradient_est_timestep2)[:, :, :, np.newaxis] * discounted_return[:, :, np.newaxis, :],
            axis=0)  # (T,K,L)
        baseline = baseline_num / (baseline_den[:, :, np.newaxis])  # (T,K,L)
        gradient = np.sum(gradient_est_timestep[:, :, :, np.newaxis] * (discounted_return[:, :, np.newaxis, :] -
                                                                baseline[np.newaxis, :, :]), axis=1) # (N,T,K,L aggiunta) * (N,T,K aggiunta,L) = (N,K,L) somma rispetto a T


        return np.mean(gradient, axis=0)