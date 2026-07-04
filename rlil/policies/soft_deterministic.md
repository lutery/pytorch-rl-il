这个方法实现了 **SAC 中对数概率的"tanh 修正"**，是整个算法最容易被忽略的细节。我分四层讲解。

---

## 第一层：为什么要"修正"？

SAC 的策略网络做了两件事：

```
raw ~ N(μ, σ)           ← 从一个高斯分布采样
action = tanh(raw)      ← 对 raw 做 tanh 压缩到 (-1, 1)
```

现在我们要算的是 `log π(action)`——**最终 action 的对数概率**，但我们只有 `log p(raw)`——**原始 raw 的对数概率**。这两个不一样。

**直觉类比**：你把一张纸（raw）揉成团（action=tanh(raw)）。纸上每一点的"密度"变了——原来均匀分布的点，揉过之后有些地方密、有些地方疏。`log π(action)` 必须补偿这个密度变化。

---

## 第二层：变 更公式

概率论里的 **变量替换公式（Change of Variables）**：

$$\log \pi(\text{action}) = \log p(\text{raw}) - \log \left| \det \frac{\partial \,\text{action}}{\partial \,\text{raw}} \right|$$

也就是：**目标分布的 log 概率 = 源分布的 log 概率 − log(雅可比行列式)**。

对于 tanh 函数，雅可比行列式有闭式解：

$$\log \left( 1 - \tanh^2(x) \right)$$

所以理论上应该写成：

```
log_prob = normal.log_prob(raw) - log(1 - tanh²(raw))
```

但**直接算 `tanh²(raw)` 数值不稳定**——当 raw 很大时，tanh(raw) → 1，`1 - tanh²(raw)` 会变得极小，产生下溢。

---

## 第三层：为什么要绕弯用 softplus？

注释里给了参考来源（OpenAI Spinning Up 的 SAC 实现），它用的恒等式是：

$$\log(1 - \tanh^2(x)) = 2 \cdot \big( \log(2) - x - \text{softplus}(-2x) \big)$$

其中 `softplus(y) = log(1 + e^y)`。

这个恒等式的价值是**数值稳定**。对比一下：

| raw 的值 | `1 - tanh²(raw)` 直接算 | `2*(log(2) - raw - softplus(-2*raw))` |
|---|---|---|
| raw → 很大 | 1 − 1² ≈ 0（下溢） | −2*raw + 常数，稳定 |
| raw → 很小 | 1 − (−1)² ≈ 0（下溢） | 0，稳定 |
| raw ≈ 0 | 1 − 0² = 1（log≈0） | 0，稳定 |

**为什么 softplus 稳定？** 当 raw 很大时，`softplus(-2*raw) → 0`，整个式子退化为 `2log(2) - 2*raw`，只是一个大负数，不会下溢。

---

## 第四层：逐行代码

```python
def compute_log_prob(self, raw, normal):
    # ① 先算 raw 在高斯分布下的 log 概率（每个维度独立，求和）
    log_prob = normal.log_prob(raw).sum(axis=-1)

    # ② 减去 tanh 变换的雅可比修正项
    log_prob -= (2 * (np.log(2) - raw - F.softplus(-2 * raw))).sum(axis=-1)

    return log_prob
```

分解一下第 ② 行的计算：

```
np.log(2)           ≈ 0.6931           常数
raw                                     原始采样值（每个维度）
F.softplus(-2*raw)                      log(1 + e^(-2*raw))

2 * (0.6931 - raw - softplus(-2*raw))   tanh 修正量（单个维度）
.sum(axis=-1)                           所有维度求和
```

最后 `.sum(axis=-1)` 的含义：**每个动作维度独立算修正量，然后求和**。因为 SAC 用的是对角高斯（各维度独立），所以总对数概率 = 各维度对数概率之和。

---

## 为什么这个函数在 SAC 中至关重要？

看 SAC agent 的损失函数（以 policy loss 为例）：

```
policy_loss = E[ α * log_prob - min(Q1, Q2) ]
```

`log_prob` 直接出现在损失里，用于**熵正则化**。如果 `log_prob` 算错了（没做 tanh 修正），梯度就是错的——模型会在错误的方向上优化探索程度。而熵正则化恰好是 SAC 相比 DDPG/TD3 的核心差异，所以这个函数虽然只有两行，却是整个算法的基石之一。

---

**一个检验理解的问题**：如果 SAC 用的不是 tanh 压缩，而是直接用 `clip(raw, -1, 1)` 硬截断，`compute_log_prob` 还需要这个修正项吗？——不需要，因为 clip 不是可微的连续变换，不存在雅可比行列式，但 clip 本身会杀死梯度，所以没人这么做。