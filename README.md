# Inverse Reinforcement Learning from a Gradient-based Learner

This repository is the official implementation of Inverse Reinforcement Learning from a Gradient-based Learner.

## Requirements

- python >= 3
- numpy >= 1.17
- torch >= 1.5.0
- baselines
- mujoco_py

Add in baselines/common/vecenv/subproc_vec_env the file subproc_vec_env.py

For MuJoCo experiments, a MuJoCo licence (https://github.com/openai/mujoco-py).

## Reproducibility

### GridWorld experiments

In order to launch the gridworld experiments it is necessary to follow these steps:
- Launch one of the learners, depending on the algorithm you want to use, specyfing the batch size, len of trajectories, number of learning steps, directory to save the trajectories. The code is inside the directory learners.
    Example: python3 learners/learner_gpomdp.py

- Launch logel.py with the following command:
    python3 --dir-output directory_where_to_save
    It is possible to specify the input directory of the trajectories and if it is necessary to do the behavioral cloning
    
### MuJoCo Experiments

We adapted the PPO implementation by Ilya Kostrikov, available at https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and the implementation by Alexis Jacq, avaiable at https://github.com/alexis-jacq/Learning_from_a_Learner

To reproduce the experiments:
1. Generate the trajectories running: python3 learner.py
2. Infer the reward for:
- LOGEL running: python3 logel.py
- LfL running: python3 lfl.py

3. Train the observer with the inferred reward running:
- for LOGEL: python3 observer_logel.py
- for LfL: python3 observer_lfl.py



