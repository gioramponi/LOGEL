"""Observer's training.
This code is an adaptation of the code of LfL for MuJoCo taken from https://github.com/alexis-jacq/Learning_from_a_Learner

The observer is trained via Proximal Policy Optimization (PPO).
This code is an adaptation of the PPO implementation taken from
https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail.
"""

import collections
import glob
import os
from arguments import get_args
from envs import make_vec_envs
from model import Policy
import numpy as np
from ppo import PPO
from storage import RolloutStorage
import torch
from utils import get_vec_normalize
from utils import update_linear_schedule

args = get_args()

num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
torch.manual_seed(args.seed)

args.save_dir = args.save_dir
args.scores_dir = args.scores_dir
args.rewards_dir = args.rewards_dir
args.policies_dir = args.policies_dir

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
#
try:
    os.makedirs(args.scores_dir)
except OSError:
    files = glob.glob(os.path.join(args.scores_dir, '*' + args.expe + '.npy'))
    for f in files:
        os.remove(f)


def main():
    device = 'cpu'
    acc_steps = []
    acc_scores = []
    torch.set_num_threads(1)

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, args.add_timestep,
                         device, False)

    # get cloned policy and recovered reward function
    policy_reward_dir = args.rewards_dir

    policy_reward = Policy(envs.observation_space.shape, envs.action_space)

    policy_reward_file_name = policy_reward_dir + '/reward_lfl_' + args.env_name + '_' + args.expe + '.pth'
    policy_reward_sd = torch.load(policy_reward_file_name)
    policy_reward.load_state_dict(policy_reward_sd)

    actor_critic = Policy(envs.observation_space.shape, envs.action_space)



    agent = PPO(actor_critic, args.clip_param, args.ppo_epoch,
                args.num_mini_batch, args.value_loss_coef, args.entropy_coef,
                lr=args.lr, eps=args.eps, max_grad_norm=args.max_grad_norm)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space, 1)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = collections.deque(maxlen=10)
    num_updates = 30
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learniobserver.pyng rate linearly
            update_linear_schedule(agent.optimizer, j, num_updates, args.lr)
            agent.clip_param = args.clip_param * (1 - j / float(num_updates))

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob = actor_critic.act(
                    rollouts.obs[step],
                    rollouts.masks[step])

            obs, _, done, infos = envs.step(action)
            if step > 1 and step % 1000 == 0:
                [True for _ in range(args.num_processes)]

            # use infered reward:
            with torch.no_grad():
                # _, reward = shapes(rollouts.obs[step], 0)
                _, action_log_probs, _ = policy_reward.evaluate_actions(
                    rollouts.obs[step], None, action)
                reward = action_log_probs

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                # r = 0
                # for key, val in info.items():
                #     if 'reward' in key:
                #         r += val
                # episode_rewards.append(r)

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                       for done_ in done])
            # print(reward)
            rollouts.insert(obs, action, action_log_prob,
                            value, reward, masks, 1)

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

        if (args.eval_interval is not None
                and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            eval_envs = make_vec_envs(args.env_name, args.seed + args.num_processes,
                                      args.num_processes, args.gamma, eval_log_dir,
                                      args.add_timestep, device, True)

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

    scores_file_name = args.scores_dir + '/observer_scores_lfl_' + args.env_name + '_' + args.expe + '.npy'
    steps_file_name = args.scores_dir + '/observer_steps_lfl_' + args.env_name + '_' +  args.expe + '.npy'
    np.save(scores_file_name, np.array(acc_scores))
    np.save(steps_file_name, np.array(acc_steps))


if __name__ == '__main__':
    main()
