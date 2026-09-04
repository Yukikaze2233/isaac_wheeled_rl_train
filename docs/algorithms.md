# 算法分支指南

五个策略分支共享一个 PPO 训练核心,每个分支只实现自己与基线的差异。
本文说明每个分支的机制、接入点与适用场景。

## 共享核心:`algorithms/ppo_base.py`

`ExtPPOLoop`(231 行)实现完整 on-policy 循环:

```text
每迭代:
  collect(24 步)  →  obs/priv/hist/actions/logp/rewards/dones/values 入 storage
                  →  GAE(γ=0.99, λ=0.95)
  update()        →  advantage 归一化 → 4 minibatch × 4 epoch:
                       surrogate(clipped) + 2·value_loss − 0.005·entropy
                       + policy.extra_loss()        ← 分支辅助损失
                       + policy.surrogate_penalty() ← 分支约束项(可选)
                     → grad clip 1.0(分布式时 all-reduce 平均)
                     → adaptive-KL 调 lr(目标 KL 0.02,×0.5 / ×1.5)
```

迭代续训:`save/load` 保存策略+优化器+迭代计数+学习率;`metrics_hook` 每迭代回调。

## 分支对照表

| 分支 | 文件 | 记忆来源 | Actor 输入 | 辅助损失 | 适用 |
|---|---|---|---|---|---|
| ppo_hist(基线) | `experiments/ppo_hist.py` | 无 | 5 帧展平历史 | 无 | 对照组,任何新机制的"是否有用"标尺 |
| HIM | `algorithms/him.py` | 历史一步特权估计 | obs + 16D latent | estimator MSE | 有真实特权流可回归时首选,结构最简单 |
| DreamWaQ | `algorithms/dreamwaq.py` | CENet-VAE 隐式编码 | obs + 16D latent | recon + KL(权重 1.0) | 特权流噪声大/维度高时,隐式估计更稳 |
| NP3O | `algorithms/np3o.py` | BarlowTwins 表征 | obs + 16D latent | twins 一致性 + cost critic MSE | 有安全/约束需求(倾角、高度、接触力限值) |
| GRU | `algorithms/recurrent.py` | GRU(64) 演化历史序列 | obs(残差直通)+ hidden | 无 | 需要长时序记忆且不想手搓历史长度 |

## 分支接入点(写新分支时只碰这三处)

```python
class MyBranch(HistoryRoller, nn.Module):
    def act(self, obs, streams):            # 1) 部署路径:编码记忆 + 采样
        ...
    def update_distribution(self, obs, hist_flat=None):   # 2) 训练路径:同编码
        ...
    def evaluate(self, priv_obs, hist_flat=None):         # 3) critic(可吃特权流)
        ...
    def extra_loss(self, obs, priv, hist, priv_hist):     # 可选:辅助损失
        ...
    def surrogate_penalty(self, obs, hist_flat, priv_obs) # 可选:约束惩罚
```

约定:
- **不要在 act 与 update_distribution 间共享可变 buffer**( latent 要按 minibatch 行
  从各自历史重算,不能复用采集期的缓存——否则 batch 维不匹配或梯度图断裂)。
- critic 吃特权流(真实速度/高度/DR 回读),actor 只吃合同内观测;两者输入维度独立。

## 各分支机制细节

### HIM(历史一步特权估计)
估计器 `hist_flat → (latent, priv_pred)`;训练目标 = 下一帧特权流(detach),
MSE 损失;actor 吃 `obs ⊕ latent`,critic 吃 `priv ⊕ latent`。
部署只需机载历史,无特权依赖。

### DreamWaQ(隐式估计)
CENet 编码器出 (μ, logσ),重参数化采 z,解码器重建特权流;
损失 = 重建 MSE + KL(N(μ,σ) ‖ N(0,1))(按特权维归一)。
与 HIM 的区别:特权信息压缩进分布而非直接回归,对特权流质量更鲁棒。

### NP3O(约束 PPO)
- 表征:两个错位历史视图过 BarlowTwins 编码器,互相关矩阵推向单位阵
  (对角=1,非对角惩罚 0.01),无负样本对。
- 约束:cost critic 回归即时违反量(来自特权流),Lagrangian 乘子按
  `λ += 0.05 × (mean_cost − limit)` 自动调,均值成本被推向 cost_limit。
- `surrogate_penalty = λ × cost.mean()` 加进 PPO 损失。

### GRU(循环记忆)
GRU 在 (N, H, obs_dim) 历史序列上演化,取末步 hidden;actor 输入 = obs ⊕ hidden
(残差设计:当前观测直通路径保留)。toy 实测收敛 +0.34,是记忆类分支里最强的基线。

## 实验化

每个分支绑定实验条目(见 `experiments/registry.py`):

```bash
python scripts/run_experiment.py --exp exp001_him_latent16 --iterations 60
python scripts/run_experiment.py --exp exp001_him_latent16 --resume   # 续训
python scripts/compare_experiments.py runs/* --csv results.csv
```

控制变量原则:对比分支时只改 branch 字段,其余 cfg/env_kwargs/seed 全同
(exp003 vs exp004 是标准范例——唯一变量是 cost limit)。

## 本机验证

```bash
PYTHONPATH=src python tests/test_algorithms.py        # 三分支收敛断言
PYTHONPATH=src python tests/test_season_features.py   # GRU 分支收敛断言
```
