from abc import ABC, abstractmethod
import numpy as np
import torch
from rlil.environments import State, Action
from rlil.initializer import get_device, is_debug_mode
from .replay_buffer import ExperienceReplayBuffer
from .base import BaseBufferWrapper
from .gae_wrapper import GaeWrapper


class SqilWrapper(BaseBufferWrapper):
    """
    构建SQIl的缓冲区
    SQIL is a behavior cloning method which regularizes the 
    reward to sparse by giving the agent a constant 
    reward of r = +1 for matching the demonstrated action in 
    a demonstrated state, and giving the agent a constant reward
    of r = 0 for all other behavior.
    https://arxiv.org/abs/1905.11108
    """

    def __init__(self, buffer, expert_buffer):
        """当然可以。SQIL 选择 0 是一种**设计决策**，不是唯一解。你的问题实际上是："如果把 0 换成别的，会发生什么？"

---

## 把问题具体化：三种替代方案

你打开着 `sqil_wrapper.py`，改的就是这两行：

```python
rewards = torch.zeros_like(rewards)          # 目前：学生 = 0
exp_rewards = torch.ones_like(exp_rewards)   # 专家 = 1
```

| 替代方案 | 代码变化 | 效果 |
|---|---|---|
| A. 保留环境原始 reward | `rewards = rewards`（不动） | 混合两个目标："做好任务" + "模仿专家" |
| B. 全部用原始 reward | 连 SqilWrapper 都不需要 | 就是普通 SAC，expert data 只是多了些高质量样本 |
| C. 学生设为 -1 | `rewards = -torch.ones_like(rewards)` | 主动惩罚"不模仿"的行为 |

---

## 方案 A：保留原始 reward（混合目标）

```python
# 学生自己的数据，reward 用环境给的
rewards = rewards   # 不覆盖
# 专家数据，reward 用 1 替换（或也用原始环境 reward？）
exp_rewards = torch.ones_like(exp_rewards)
```

**好处**：智能体既能学会任务（比如让机器人走起来），又会被引导向专家风格。

**问题**：两个目标可能冲突。假设你在训练一个 Walker2D 向前走——环境 reward 是"向前的速度 + 能耗惩罚"。专家走得很潇洒但不一定最快。如果保留环境 reward，SAC 可能发现"大步前倾比专家的优雅小步更快"，从而偏离模仿目标。

**适用场景**：你想让 agent 在**新环境**（和专家不同 MDP，如重力减半）里既适应环境又参考专家。本文库中的 `rlil/environments/` 里的变重力环境就是为此设计的。

---

## 方案 B：完全不用 SQIL（纯数据增强）

就是把 expert data 扔进 replay buffer，不做任何 reward 修改。等价于：

```python
# 不用 SqilWrapper，直接用 base_agent 的 buffer
expert_replay_buffer.store(expert_samples)  # 直接混进普通 buffer
```

**好处**：最简单。

**问题**：专家的数据里 reward 是**专家轨迹的累积 reward**，包含了未来步的贡献。如果专家的 reward 很高（比如专家在 Ant 任务里每步 reward=30），而学生刚开始每步 reward=2，混在一起会造成数据分布严重不均衡。Q 函数会被专家数据的高 reward 淹没，学到的不是"怎么模仿"，而是"专家数据里的高 reward 是什么状态给的"。

---

## 方案 C：学生设为负数（惩罚）

```python
rewards = -torch.ones_like(rewards)   # 学生 = -1
```

**好处**：更强地"推开"非专家行为，理论收敛更快。

**问题**：
1. 太强的负 reward 会让智能体**不敢探索**——一旦它发现自己做的行为都是 -1，可能选择"不动"来最小化损失
2. SAC 的熵正则化会和负 reward 形成拉扯，可能导致训练不稳定
3. 在连续动作空间中，-1 惩罚太"绝对"——专家附近的动作（和专家相差 0.01）也被判 -1，不合理

**论文原文的实验结论**：SQIL 的作者确实试过 -1/0/1 的设定，但 0/1 效果更好。原因是 0 给智能体足够的"容错空间"去探索状态空间，而 -1 让智能体过度保守。

---

## 为什么 SQIL 坚持用 0？——两个理论支撑

### 支撑一：0 的"吸收态"性质

在 soft Q-learning 的 Bellman equation 里：

```
Q(s,a) = r + γ · V(s')
```

如果 r=0，那么 `Q(s,a) = γ · V(s')`，Q 值完全来自 **下一状态的值**。这意味着 0 不创造也不消灭信息——它只是把价值传递往后推了一格。这让 Bellman 传播**干净**：正的 Q 值只能来自"最终碰到了专家状态"，不会因为"中间某步做了件任务层面正确的事"而产生假阳性信号。

### 支撑二：0 是 SQIL 理论推导的一部分

SQIL 的 reward 设计是为了让 SAC 的目标等价于：

$$\pi^* = \arg\min_\pi \; D_{KL}(\rho^\pi \parallel \rho^E)$$

这个等价性**依赖** expert reward=1 且 agent reward=0。如果你改变了 agent reward，上面的等价就不再成立，SQIL 就退化成了某种 reward shaping + RL，而不是一个有理论保证的模仿学习方法。

---

## 实践中的灵活处理

真实使用时，很多人会做**渐进式**处理：

```
阶段 1（前 30% 训练）：  学生 reward=0，只用 0/1 信号，快速收敛到专家分布
阶段 2（后 70% 训练）：  逐步混入环境原始 reward，让策略适应具体任务
```

或者加一个**混合系数** λ：

```python
agent_reward = λ * original_reward + (1-λ) * 0
expert_reward = 1
```

λ 从 0 逐渐增大到 1，实现"先模仿后优化"。但这就不是纯 SQIL 了，是 SQIL + 环境 reward 的组合方案。
        Args:
            buffer (rlil.memory.ExperienceReplayBuffer): 
                A replay_buffer for sampling.
            expert_buffer (rlil.memory.ExperienceReplayBuffer):
                A replay_buffer with expert trajectories.
        """
        self.buffer = buffer
        self.expert_buffer = expert_buffer
        self.device = get_device()

    def sample(self, batch_size):
        batch_size = int(batch_size / 2)
        states, actions, rewards, next_states, weights, indexes = \
            self.buffer.sample(batch_size)
        exp_states, exp_actions, exp_rewards, exp_next_states, \
            exp_weights, exp_indexes = self.expert_buffer.sample(batch_size)

        rewards = torch.zeros_like(rewards, dtype=torch.float32,
                                   device=self.device)
        exp_rewards = torch.ones_like(exp_rewards, dtype=torch.float32,
                                      device=self.device)

        states = State.from_list([states, exp_states])
        actions = Action.from_list([actions, exp_actions])
        rewards = torch.cat([rewards, exp_rewards], axis=0)
        next_states = State.from_list([next_states, exp_next_states])
        weights = torch.cat([weights, exp_weights], axis=0)

        # shuffle tensors
        index = torch.randperm(len(rewards))
        if indexes is None or exp_indexes is None:
            indexes = None
        else:
            indexes = torch.cat([indexes, exp_indexes], axis=0)[index]

        return (states[index],
                actions[index],
                rewards[index],
                next_states[index],
                weights[index],
                indexes)
