"""
Function to create trajectories following an argmax policy.
"""

import numpy as np

def create_batch_trajectories(pi, batch_size, len_trajectories, env, rew_param=25, render=False):
    trajectories = np.zeros((batch_size,len_trajectories, 2))
    rewards = np.zeros((batch_size, len_trajectories, 1))
    feat_rewards = np.zeros((batch_size, len_trajectories, rew_param))
    state = env.reset()
    for n in range(batch_size):
        env.reset()
        for t in range(len_trajectories):
            state_int = int(env._coupleToInt(state[0], state[1]))
            a = int(pi[state_int])
            trajectories[n,t] = (int(state_int), int(a))
            state, rew, feat = env.step(state, a)
            rewards[n,t] = rew
            feat_rewards[n,t] = feat
    return trajectories, rewards, feat_rewards