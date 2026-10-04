# v13 冻结与接续回执

当前：本机冻结、CPU 验证和 30 角色物化检查完成；Kaiser SSH 无路由，013b 原生验证和 PPO **尚未启动**。

- 冻结源码：`7ab0a329c3703d6e71a4dc6a7dbc8dfb8ddbc685`。
- 源码归档 SHA：`44e939f09ad2149d8e995970374f743d26ad398f868fc48685dc4d5687d4fa77`。
- v13 计划 SHA：`a82b249a1a4647b8291ba031a7cb85d422fdde65cb25cd9ad919616e51ae13ec`。
- 首份仅出生候选 013 未运行原生场景或 PPO，被松键停车补强替代；原始归档和回执保留。
- 旧 jump 已合法封存 task actor 2735／消费 23868，完整检查点 SHA `d0579a3f58965ffc867668e34d51043386cb6fb7fd99ea270f9d6967eea8621b`，见 `nonstandard_seal_013.json`。

相关 CPU 集合已通过，数字重叠，不能相加：迁移与独立门禁 62、出生／mask／wrapper 集合 118、停车／姿态 hooks 集合 51、扩展地形与停车控制 36；历史 SCUT 技能列表断言仍单列，不伪装成新通过。测试临时目录配额失败的用例在独立工作目录重跑通过。

新实现保留 1.5 m/s²／4 rad/s²；新增 5% 松键停车池、0.2 s 零主动力矩的合法闭链出生、加速／恒速／减速／停车窗的条件姿态指标与样本数。terrain 仅新停车组扩展原地形至 40 m。奖励不改，旧评测关闭新教学，四个其他 worker 与预算不改；腿 bag 未 qualified，仍不接入。

独立核查的 50 项聚焦测试通过。吞吐尚未原生验证：rough 场地扩展为 105 个表面可能增加整个 terrain 批次的补齐维度及查询开销，需在原生验证和续训时测量，不能承诺耗时不变。

恢复网络后的执行顺序：

1. 上传 `source_candidate_013b.json` 指定的不可变归档，运行本目录 `prepare_revision_013b.py`；校验整归档、同 root 迁移和远端 30 角色。
2. 运行 `validate_native_013b.py`，先 speed 再 terrain，各 4096 环境、10 s 物理时间；使用远端 runtime bootstrap。不建优化器，不计 PPO 消费，不把原生接入通过当停车质量通过。
3. 只有两份原生回执通过且无同 root 存活 worker 时，运行 `launch_frozen_013b.py`。它启动 tmux r13 和唯一原生监督进程，使用同 root `--resume-existing` 与不可变源码迁移回执。
4. 核对 actor／critic／Adam 恢复、计费 critic 适应、首个真实 actor 参数改变、TensorBoard 出生 mask／停车相位／姿态以及实际摩擦域。更新主 launch 回执后再描述为正在学习。

自动顺序是 speed 自己的 3563→5063、terrain 自己的 1186→2186，然后同合同恢复 jump 2735 及后续 stepjump。原消费不退款，无 45 项阻断 gate。源码归档后仅提交结果文档不触发重启。
