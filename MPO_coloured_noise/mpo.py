import gymnasium as gym
from tonic import environments, Trainer
import tonic
from pink import MPO_CN
import d4rl
from tonic.torch.agents import MPO
import shimmy

# from pink.cnrl import MonitorCallback
# Initialize environment

# envs = ['MountainCarContinuous-v0']
# env_name = 'MountainCarContinuous-v0'
env_names = [   
                'pendulum-swingup', 
                'cartpole-balance_sparse',
                'cartpole-swingup_sparse',
                'ball_in_cup-catch',
                'reacher-hard',
                'hopper-hop',
                'cheetah-run',
                'walker-run'
            ]

seed = 42
# set seed for np and torch and random
# set seed for gym envs
# set seed for d4rl
# change seed for random
import random
random.seed(seed)
# change seed for torch
import torch
torch.manual_seed(seed)


for env_name in env_names:
    _env =f"tonic.environments.ControlSuite('{env_name}')"
    env = tonic.environments.distribute(
        lambda: eval(_env), 1, 1)
    seq_len = env.max_episode_steps

    for beta in range(3):
        model = MPO_CN()
        model.initialize(beta, seq_len, env.observation_space, env.action_space)
        # Train agent
        trainer = Trainer(steps=1e6)
        trainer.initialize(model, env, env_name="./res/"+env_name+f"_PinkNoise_{seed}")
        trainer.run()
        del model, trainer
