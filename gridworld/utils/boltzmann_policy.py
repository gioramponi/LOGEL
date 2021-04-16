"""
Functions used to generate trajectories following the Boltzmann policy
"""
import numpy as np

"""
Function to create a batch of trajectories using the boltzmann policy having phi * theta on the exponent of the boltzmann
"""
def create_batch_trajectories(pi, batch_size, len_trajectories, env, rew_param=25, render=False):
    # trajectories = np.zeros((batch_size, len_trajectories, 2))
    trajectories = np.zeros((batch_size, len_trajectories, 2))
    rewards = np.zeros((batch_size, len_trajectories, 1))
    feat_rewards = np.zeros((batch_size, len_trajectories, rew_param))
    state = env.reset()
    for n in range(batch_size):
        env.reset()
        for t in range(len_trajectories):
            state_int = int(env._coupleToInt(state[0], state[1]))
            probs = np.array([np.dot(env.phi(state_int, act), pi) for act in range(3)]).ravel()
            probs = np.concatenate((probs, np.array([0])))
            probs -= np.max(probs)
            probs = np.exp(probs)
            probs /= np.sum(probs)
            a = int(np.random.choice(np.arange(4), p=probs))
            trajectories[n, t] = (int(state_int), int(a))
            state, rew, feat = env.step(state, a)  # , render=render)
            rewards[n, t] = rew
            feat_rewards[n, t] = feat
    return trajectories, rewards, feat_rewards


"""
Function to create a batch of trajectories using the boltzmann policy wth Q on the exponent of the boltzmann
"""
def create_batch_trajectories_with_Q(Q, batch_size, len_trajectories, env, rew_param, render=False):
    trajectories = np.zeros((batch_size * len_trajectories, 2))
    rewards = np.zeros((batch_size, len_trajectories, 1))
    feat_rewards = np.zeros((batch_size, len_trajectories, rew_param))
    state = env.reset()
    for n in range(batch_size):
        env.reset()
        for t in range(len_trajectories):
            state_int = int(env._coupleToInt(state[0], state[1]))
            probs = np.array([Q[act, state_int] for act in range(4)]).ravel()
            probs -= np.max(probs)
            probs = np.exp(probs)
            probs /= np.sum(probs)
            a = int(np.random.choice(np.arange(4), p=probs))
            trajectories[n * len_trajectories + t] = (int(state_int), int(a))
            state, rew, feat = env.step(state, a)  # , render=render)
            rewards[n, t] = rew
            feat_rewards[n, t] = feat
    return trajectories, rewards, feat_rewards


"""
Compute the gradient of a given batch of trajectories
"""


def compute_gradients(batch_size, len_trajectories, trajectories, env, pi):
    gradients = np.zeros((batch_size, len_trajectories, pi.shape[0]))
    for n in range(batch_size):
        for t in range(len_trajectories):
            s = int(trajectories[n, t, 0])
            a = int(trajectories[n, t, 1])
            gradients[n, t] = get_gradient(s, a, env, pi)
    return gradients


"""
Function to compute the gradient given the environment and the parameters of the current policy
"""
def get_gradient(s, a, env, pi):
    values = np.array([np.dot(env.phi(s, i), pi) for i in range(3)])
    values2 = np.array([env.phi(s, i) for i in range(3)])
    maxx = np.max(values)
    values -= np.max(values)
    values3 = np.exp(values) * values2
    if a == 3:
        return -np.sum(values3, axis=0) / (np.sum(np.exp(values)) + np.exp(-maxx))
    return values2[a] - np.sum(values3, axis=0) / (np.sum(np.exp(values)) + np.exp(-maxx))


def compute_trajectories(tr_len, n_batch, pi, env):
    trajectories = []
    state = env.reset()
    for b in range(n_batch):
        state_int = env._coupleToInt(state[0], state[1])
        for t in range(tr_len):
            probs = np.array([np.dot(env.phi(state_int, a), pi) for a in range(4)]).ravel()
            probs -= np.max(probs)
            probs = np.exp(probs)
            probs /= np.sum(probs)
            act = np.random.choice(np.arange(4), p=probs)
            trajectories.append((state_int, act))
            state_x, state_y = env.transition(state, act)
            state = (state_x, state_y)
            state_int = env._coupleToInt(state[0], state[1])
    return np.array(trajectories)


"""
Function to compute the reward and the feature vector given a batch of trajectories
"""
def compute_reward_and_feat(trajectories, n_batch, tr_len, env, render=False):
    n_batch = n_batch
    tr_len = tr_len
    rewards = np.zeros((n_batch, tr_len, 1))
    features = np.zeros((n_batch, tr_len, env.reward_weights.shape[0]))
    for b in range(n_batch):
        rew = np.zeros((tr_len, 1))
        feats = np.zeros((tr_len, env.reward_weights.shape[0]))
        for tr in range(tr_len):
            state = trajectories[b, tr, 0]
            act = trajectories[b, tr, 1]
            r = env.get_reward(state)
            f = env.get_reward_vector(state)
            rew[tr] = r
            feats[tr] = f
        rewards[b] = rew
        features[b] = feats
    return rewards, features


"""
Function to compute the variance of the maximum likelihood of a set of given trajectories
"""
def variance_maximum_likelihood(trajectories, env, mean_mle):
    variance = 0  # np.zeros((mean_mle.shape[0], mean_mle.shape[0]))
    for s, a in trajectories:
        s = int(s)
        values = np.array([np.dot(env.phi(s, i), mean_mle) for i in range(3)])
        values2 = np.array([env.phi(s, i) for i in range(3)])
        prob_values = np.exp(values) / (np.sum(np.exp(values)) + 1)
        exp_feat = np.sum(prob_values * values2, axis=0)
        out_prod = np.array([np.outer(env.phi(s, i), env.phi(s, i)) for i in range(3)])
        ex_out_prod = np.sum(prob_values[:, :, np.newaxis] * out_prod, axis=0)
        variance += np.outer(exp_feat, exp_feat) - ex_out_prod
    try:
        return - np.linalg.inv(variance)
    except:
        # print('yes')
        return np.linalg.inv(-variance + 0.1 * np.identity((variance.shape[0])))
