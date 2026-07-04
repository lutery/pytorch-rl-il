'''
Pytorch models for continuous control.
'''
import numpy as np
import torch
from rlil import nn


def fc_q(env, hidden1=400, hidden2=300):
    '''
    全连接层预测Q值
    看来这里是环境的维度和动作维度，看来是通过环境的状态和动作来预测Q值

    env: 环境
    hidden1、hidden2: 隐藏层的维度
    '''

    return nn.Sequential(
        nn.Linear(env.state_space.shape[0] +
                  env.action_space.shape[0], hidden1),
        nn.LeakyReLU(),
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, 1),
    )


def fc_v(env, hidden1=400, hidden2=300):
    '''
    状态价值预测网络
    
    '''
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0], hidden1),
        nn.LeakyReLU(),
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, 1),
    )


def fc_deterministic_policy(env, hidden1=400, hidden2=300):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0], hidden1),
        nn.LeakyReLU(),
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, env.action_space.shape[0]),
    )


def fc_deterministic_noisy_policy(env, hidden1=400, hidden2=300):
    return nn.Sequential(
        nn.NoisyFactorizedLinear(env.state_space.shape[0], hidden1),
        nn.LeakyReLU(),
        nn.NoisyFactorizedLinear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.NoisyFactorizedLinear(hidden2, env.action_space.shape[0]),
    )


def fc_soft_policy(env, hidden1=400, hidden2=300):
    '''
    全连接网络，根据环境状态预测动作的均值和 log(σ²) 
    '''
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0], hidden1),
        nn.LeakyReLU(),
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, env.action_space.shape[0] * 2),
    )


def fc_actor_critic(env, hidden1=400, hidden2=300):
    features = nn.Sequential(
        nn.Linear(env.state_space.shape[0], hidden1),
        nn.LeakyReLU(),
    )

    v = nn.Sequential(
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, 1)
    )

    policy = nn.Sequential(
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, env.action_space.shape[0] * 2)
    )

    return features, v, policy


def fc_discriminator(env, hidden1=400, hidden2=300):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0] + env.action_space.shape[0],
                  hidden1),
        nn.LeakyReLU(),
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, 1),
        nn.Sigmoid())


def fc_bcq_encoder(env, latent_dim=32, hidden1=400, hidden2=300):
    # output mean and log_var
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0] +
                  env.action_space.shape[0], hidden1),
        nn.LeakyReLU(),
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, latent_dim * 2)
    )


def fc_bcq_decoder(env, latent_dim=32, hidden1=300, hidden2=400):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0] + latent_dim, hidden1),
        nn.LeakyReLU(),
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, env.action_space.shape[0])
    )


def fc_bcq_deterministic_policy(env, hidden1=400, hidden2=300):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0] +
                  env.action_space.shape[0], hidden1),
        nn.LeakyReLU(),
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, env.action_space.shape[0]),
    )


def fc_reward(env, hidden1=400, hidden2=300):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0] +
                  env.action_space.shape[0], hidden1),
        nn.LeakyReLU(),
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, 1)
    )


def fc_dynamics(env, hidden1=500, hidden2=500):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0] +
                  env.action_space.shape[0], hidden1),
        nn.LeakyReLU(),
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, env.state_space.shape[0]),
    )
