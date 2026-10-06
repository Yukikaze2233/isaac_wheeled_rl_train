# source023 固定窗口只读观察

默认只输出远端stdin Python代码，不执行SSH、不写远端文件、不创建训练或评测。既有四个reader与实际023 receipt/source身份按SHA引用；生产PPO不依赖本工具。

本地emit：

```sh
python -B reports/v6_scut_posture_completion_20261006/source_repair_023/read_recovery_window_023.py --end-actor 110
```

输出交给Kaiser既有 `/home/kaiser/robot-rl-sim60/env/bin/python -B -` stdin，stdout保存为独立观察回执。后续端点210/310，对应不重叠111–210/211–310；不能替换为latest滑窗。首窗REC6436–6535/TB33640–33739，真实Behavior基线33639，前10critic及无首基线actor前段排除但消费照计。

实际pending001：40actor可用，差70，exit0；PIDs/identity连续，没有解析未齐event。语法/边界自检通过不证明完整解析路径实际执行。原生脚本控制样本排除、当前采样统计、L2整链与确定性部署资格分别解释；缺tags返回pending，不能补0。读失败仅为观察失败，不能因此重启训练。

根第一次调用误用旧CLI的--end-update在本机argparse拒绝，未SSH；改为本工具--end-actor后实际远端只读通过。
