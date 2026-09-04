# 实验工作流

控制变量实验的定义、运行、续训与对比。工具链:`experiments/registry.py` +
`scripts/run_experiment.py` + `scripts/compare_experiments.py`。

## 1. 定义一条实验

在 `experiments/registry.py` 注册:

```python
register(ExperimentSpec(
    name="exp007_my_branch_v1",          # 全局唯一
    branch="him",                        # ppo_hist | him | dreamwaq | np3o | gru
    description="...",                   # 一句话说明变量
    overrides={},                        # ExtTrainCfg 覆盖项(可选)
    env_kwargs={"hist_len": 3},          # toy 环境参数(本机模式)
    isaaclab_task="WheeledBiped-Flat-v0",# 服务器任务 id
    isaaclab_note="...",                 # 服务器集成说明(runner_class 等)
))
```

**控制变量纪律**:一组对比实验之间只允许一个字段不同。现役实验组:

| 组 | 实验 | 唯一变量 |
|---|---|---|
| 记忆形态 | exp000(frame-stack) / exp001(HIM) / exp002(DreamWaQ) / exp006(GRU) | branch |
| 约束松紧 | exp003(limit 0.5) / exp004(limit 1.5) | cost_limit |
| 任务域 | exp005(Rough + 课程) | isaaclab_task |

## 2. 运行

```bash
# 列出全部实验
python scripts/run_experiment.py --list

# 本机 toy 模式(CPU,任意机器)—— metrics 逐迭代写 runs/<exp>/metrics.jsonl
python scripts/run_experiment.py --exp exp001_him_latent16 --iterations 60

# 断点续训(策略+优化器+迭代计数+学习率全恢复)
python scripts/run_experiment.py --exp exp001_him_latent16 --resume

# 服务器模式:打印对应的 Isaac Lab 训练命令与集成说明(不本地跑)
python scripts/run_experiment.py --exp exp005_rough_curriculum --backend isaaclab
```

## 3. 产物

```text
runs/<exp_name>/
├── checkpoint.pt     # policy + optimizer + iteration + lr
└── metrics.jsonl     # {"iteration", "ep_reward", "extra", "kl", "lr"} 每迭代一行
```

## 4. 对比

```bash
python scripts/compare_experiments.py runs/exp001* runs/exp002*            # 终端表
python scripts/compare_experiments.py runs/* --csv results.csv --tail 20  # CSV 导出
```

输出按尾窗(默认 10 行)均值 ep_reward 排序,同时给出辅助损失与 KL 均值——
**读表顺序**:先看 ep_reward 差距,再用 extra 判断辅助机制是否仍在学习(持续高值
= 未收敛或冲突),KL 异常大说明该分支不稳定而非不优。

## 5. 服务器全规模流程

```bash
# 1) 按实验 spec 打印服务器命令
python scripts/run_experiment.py --exp exp001_him_latent16 --backend isaaclab

# 2) 在 Isaac Lab 环境执行(4096 envs × 20k iterations)
./isaaclab.sh -p scripts/train.py --task WheeledBiped-Flat-v0 \
    --num_envs 4096 --max_iterations 20000 --headless --log_dir runs/server_exp001

# 3) checkpoint 回放验证
./isaaclab.sh -p scripts/play.py --checkpoint runs/server_exp001/model_final.pt

# 4) 导出 ONNX 并过合同校验
./isaaclab.sh -p scripts/export_onnx.py --checkpoint runs/server_exp001/model_final.pt \
    --output policy.onnx
python3 ../isaac_wheeled_rl_deploy/tools/check_onnx_contract.py policy.onnx
```

续训:`--checkpoint runs/server_exp001/model_latest.pt` 传给 train.py
(从 checkpoint 续训可把调参闭环从数小时缩到 ~1h)。

## 6. 迭代纪律

1. 改 cfg/registry,不改共享循环。
2. 本机 toy 先跑通(分钟级),再上服务器(小时级)。
3. 每次上服务器前全量回归:`PYTHONPATH=src python tests/test_mdp.py test_env_features.py test_algorithms.py test_rsl_rl_smoke.py test_experiments.py`。
4. 对比结论写回 registry 的 description——实验史是仓库的一部分。
