"""Learner's training.
This code is an adaptation of the code of LfL for MuJoCo taken from https://github.com/alexis-jacq/Learning_from_a_Learner


The learner is trained via Proximal Policy Optimization (PPO).
This code is an adaptation of the PPO implementation taken from
https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail.
"""

import collections
import glob
import os
from arguments import get_args
from envs import make_vec_envs
from model import Policy
from rbf import *
import numpy as np
from ppo import PPO
from storage import RolloutStorage
import torch
from utils import get_vec_normalize
from utils import update_linear_schedule
import tensorflow

args = get_args()

print(args.env_name)
num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
print(num_updates)
torch.manual_seed(args.seed)


args.demos_dir = args.demos_dir
demos_expe_dir = args.demos_dir + '/expe_'+ str(args.expe) + '_' + str(args.env_name)
args.save_dir = args.save_dir
args.scores_dir = args.scores_dir


try:
    os.makedirs(args.demos_dir)
except OSError:
    files = glob.glob(os.path.join(args.demos_dir, '*.npy'))
    for f in files:
        os.remove(f)

try:
    os.makedirs(args.scores_dir)
except OSError:
    files = glob.glob(os.path.join(args.scores_dir, '*' + args.expe + '.npy'))
    for f in files:
        os.remove(f)

try:
    os.makedirs(args.log_dir)
except OSError:
    files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)

eval_log_dir = args.log_dir + '_eval'

try:
    os.makedirs(eval_log_dir)
except OSError:
    files = glob.glob(os.path.join(eval_log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)


try:
    os.makedirs(demos_expe_dir)
except OSError:
    files = glob.glob(os.path.join(demos_expe_dir, '*.npy'))
    for f in files:
        os.remove(f)


def main():
    device = 'cpu'
    acc_steps = []
    acc_scores = []
    torch.set_num_threads(1)
    print('here')

    if args.env_name == 'Reacher-v2':
        rbf1 = build_features_reacher2(.2, 5, 2)
        len_rbf = rbf1._K
        len_features = len_rbf + 1
    if args.env_name == 'Hopper-v2':
        len_features = 3
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, args.add_timestep,
                         device, False)

    actor_critic = Policy(envs.observation_space.shape, envs.action_space)

    actor_critic.to(device)

    agent = PPO(actor_critic, args.clip_param, args.ppo_epoch,
                args.num_mini_batch, args.value_loss_coef, args.entropy_coef,
                lr=args.lr, eps=args.eps, max_grad_norm=args.max_grad_norm)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space, len_features)
    print('here2')
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)
    episode_rewards = collections.deque(maxlen=10)
    num_updates = 20
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            update_linear_schedule(agent.optimizer, j, num_updates, args.lr)
            agent.clip_param = args.clip_param * (1 - j / float(num_updates))

        # Prepare demos
        demo_actions = np.zeros((1, args.num_processes, envs.action_space.shape[0]))
        demo_states = np.zeros((1, args.num_processes,
                                envs.observation_space.shape[0]))

        demo_features = np.zeros((1, args.num_processes, len_features))
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob = actor_critic.act(
                    rollouts.obs[step],
                    rollouts.masks[step])

            # obs, reward and next obs
            demo_actions = np.concatenate(
                [demo_actions, action.reshape(1, args.num_processes, -1)], 0)
            demo_states = np.concatenate(
                [demo_states, rollouts.obs[step].reshape(1, args.num_processes, -1)],
                0)
            feat_rewards = np.zeros((args.num_processes, len_features))
            if args.env_name == 'Hopper-v2':
                if args.num_processes > 1:
                    pos_before = envs.get_sim_data()
            obs, reward, done, infos = envs.step(action)
            if args.env_name == 'Hopper-v2':
                if args.num_processes > 1:
                    pos_after = envs.get_sim_data()
                    for num_p in range(args.num_processes):
                        feat_1 = pos_after[num_p] - pos_before[num_p]
                        feat_2 = 0
                        if not done[num_p]:
                            feat_2 = 1
                        # feat_2 = np.array([1 for _ in range(args.num_processes)])
                        feat_3 = np.array([np.linalg.norm(action[num_p], ord=2)**2]).flatten()
                        feat_rewards[num_p] = np.array([feat_1, feat_2, feat_3])
            if args.env_name == 'Reacher-v2':
                if args.num_processes > 1:
                    body_data = envs.get_body_data()
                    for num_p in range(args.num_processes):
                        rbf1_ = rbf1(body_data[num_p][:-1])
                        rbf4_ = np.array([np.linalg.norm(action[num_p], ord=2)**2])
                        feat_rewards[num_p] = np.concatenate((rbf1_.reshape(-1), rbf4_))
                else:
                    rbf1_ = rbf1((envs.envs[0].env.env.get_body_com("fingertip") - envs.envs[0].env.env.get_body_com("target"))[:-1])
                    rbf4_ = np.array([-np.square(action[0]).sum()])
                    feat_rewards[0] = np.concatenate((rbf1_.reshape(-1), rbf4_))
            demo_features = np.concatenate(
                [demo_features, feat_rewards.reshape(1, args.num_processes, -1)],
                0)
            if step > 1 and step % 1000 == 0:
                done = [True for _ in range(args.num_processes)]

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            rollouts.insert(obs, action, action_log_prob, \
                            value, reward, masks, feat_rewards)

        # Save demos:
        action_file_name = demos_expe_dir + '/actions_step_' + str(j) + '.npy'
        state_file_name = demos_expe_dir + '/states_step_' + str(j) + '.npy'
        rew_feat_file_name = demos_expe_dir + '/rew_feat_step_' + str(j) + '.npy'
        policy_file_name = demos_expe_dir + '/policy_step_' + str(j) + '.pth'
        np.save(action_file_name, demo_actions)
        np.save(state_file_name, demo_states)
        np.save(rew_feat_file_name, demo_features)
        torch.save(actor_critic.state_dict(), policy_file_name)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.gamma, args.tau)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0 or j == num_updates - 1) and args.save_dir:
            save_path = os.path.join(args.save_dir, 'ppo')
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor_critic

            save_model = [save_model,
                          getattr(get_vec_normalize(envs), 'ob_rms', None)]

            torch.save(save_model, os.path.join(save_path, args.env_name + '.pt'))

        total_num_steps = (j + 1) * args.num_processes * args.num_steps

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            print('Updates', j,
                  'num timesteps', len(episode_rewards),
                  '\n Last training episodes: mean/median reward',
                  '{:.1f}'.format(np.mean(episode_rewards)),
                  '/{:.1f}'.format(np.median(episode_rewards)),
                  'min/max reward',
                  '{:.1f}'.format(np.min(episode_rewards)),
                  '/{:.1f}'.format(np.max(episode_rewards)),
                  'dist entropy', dist_entropy,
                  'value loss', value_loss,
                  'action loss', action_loss)

        if len(episode_rewards) > 1:
            acc_steps.append(total_num_steps)
            acc_scores.append(np.mean(episode_rewards))
            #print(acc_scores)

        if (args.eval_interval is not None
                and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            eval_envs = make_vec_envs(
                args.env_name, args.seed + args.num_processes, args.num_processes,
                args.gamma, eval_log_dir, args.add_timestep, device, True)

            vec_norm = get_vec_normalize(eval_envs)
            if vec_norm is not None:
                vec_norm.eval()
                vec_norm.ob_rms = get_vec_normalize(envs).ob_rms

            eval_episode_rewards = []

            obs = eval_envs.reset()
            eval_masks = torch.zeros(args.num_processes, 1, device=device)

            while len(eval_episode_rewards) < 10:
                with torch.no_grad():
                    _, action, _ = actor_critic.act(
                        obs, eval_masks, deterministic=True)

                # Obser reward and next obs
                obs, reward, done, infos = eval_envs.step(action)
                eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                                for done_ in done])
                for info in infos:
                    if 'episode' in info.keys():
                        eval_episode_rewards.append(info['episode']['r'])

            eval_envs.close()

            print('Evaluation using',
                  len(eval_episode_rewards),
                  'episodes: mean reward',
                  '{:.5f}\n'.format(np.mean(eval_episode_rewards)))

    scores_file_name = args.scores_dir + '/learner_scores_' + args.env_name + '_' + args.expe + '.npy'
    steps_file_name = args.scores_dir + '/learner_steps_' + args.env_name + '_' + args.expe + '.npy'
    np.save(scores_file_name, np.array(acc_scores))
    np.save(steps_file_name, np.array(acc_steps))


if __name__ == '__main__':
    main()
