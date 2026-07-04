import torch
import numpy as np
import torch.nn.functional as F
from rlil.approximation import Approximation
from rlil.nn import RLNetwork
from rlil.environments import squash_action


class SoftDeterministicPolicy(Approximation):
    def __init__(
            self,
            model,
            optimizer,
            space,
            name="policy",
            **kwargs
    ):
        model = SoftDeterministicPolicyNetwork(model, space)
        super().__init__(model, optimizer, name=name, **kwargs)

    def sample_multiple(self, states, num_sample=10):
        return self.model.sample_multiple(states, num_sample)

    def compute_log_prob(self, raw, normal):
        return self.model.compute_log_prob(raw, normal)

    def mean_logvar(self, states):
        return self.model.mean_logvar(states)


class SoftDeterministicPolicyNetwork(RLNetwork):
    def __init__(self, model, space):
        '''
        传入动作策略模型和动作空间的信息
        
        '''
        super().__init__(model)
        self._action_dim = space.shape[0] # 动作的维度
        self._tanh_scale = torch.tensor(
            (space.high - space.low) / 2,
            dtype=torch.float32, device=self.device) # 动作空间的范围，
        self._tanh_mean = torch.tensor(
            (space.high + space.low) / 2,
            dtype=torch.float32, device=self.device) # 原始动作空间的均值

    def forward(self, state, return_mean=False):
        # 返回预测的动作以及
        outputs = super().forward(state) # 返回预测的动作均值和方差
        if return_mean: # 如果是仅返回均值，则从最后一个维度切出动作的均值即可
            means = outputs[:, 0: self._action_dim]
            means = squash_action(means, self._tanh_scale, self._tanh_mean) # 将预测的值缩放、偏移到真实的动作空间
            return means

        # make normal distribution
        means = outputs[:, 0: self._action_dim] # 切出均值
        logvars = outputs[:, self._action_dim:] #切出方差的log，这个主要是为了避免方差为负数
        std = logvars.mul(0.5).exp_() # 因为std = √(σ²) = exp( log(σ²) / 2 )，所以模型预测的是一个log的方差平方，所以要乘以0.5
        normal = torch.distributions.normal.Normal(means, std)

        # sample from the normal distribution
        raw = normal.rsample() # 采样动作
        log_prob = self.compute_log_prob(raw, normal) # 这里是计算一个连续动作的概率log值，但是由于最终的动作会使用squash_action进行缩放和偏移，所以需要
        # 使用squash_action对动作概率log值进行纠正

        action = squash_action(raw, self._tanh_scale, self._tanh_mean)
        return action, log_prob

    def sample_multiple(self, state, num_sample=10):
        # this function is used in BEAR and BRAC training
        outputs = super().forward(state)

        # make normal distribution
        means = outputs[:, 0: self._action_dim]
        repeated_means = torch.repeat_interleave(
            means.unsqueeze(1), num_sample, 1)
        logvars = outputs[:, self._action_dim:]
        repeated_logvars = torch.repeat_interleave(
            logvars.unsqueeze(1), num_sample, 1)
        repeated_std = repeated_logvars.mul(0.5).exp_()
        # batch x num_sample x d
        normal = torch.distributions.normal.Normal(
            repeated_means, repeated_std)
        raw = normal.rsample()
        action = squash_action(raw, self._tanh_scale, self._tanh_mean)
        return action, raw

    def compute_log_prob(self, raw, normal):
        '''
        raw: 采样动作
        normal：样本采样器

        SAC 中对数概率的"tanh 修正" 看md文件
        '''
        # see openai spinningup for log_prob computation:
        # https://github.com/openai/spinningup/blob/e76f3cc1dfbf94fe052a36082dbd724682f0e8fd/spinup/algos/pytorch/sac/core.py#L53

        log_prob = normal.log_prob(raw).sum(axis=-1)
        log_prob -= (2*(np.log(2) - raw - F.softplus(-2*raw))).sum(axis=-1)
        return log_prob

    def mean_logvar(self, state):
        outputs = super().forward(state)
        means = outputs[:, 0: self._action_dim]
        logvars = outputs[:, self._action_dim:]
        return means, logvars

    def to(self, device):
        self._tanh_mean = self._tanh_mean.to(device)
        self._tanh_scale = self._tanh_scale.to(device)
        return super().to(device)
