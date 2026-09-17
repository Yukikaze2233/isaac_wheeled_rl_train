# V5 第一阶段研究训练

入口：`scripts/train_chassis.py --contract contracts/v5_foundation_v1.json`。
该合同使用 V5 的19刚体、18树轴、6闭环、两套10 MPa气簧；主动轴是四个根部输出轴和两轮。
保留源质量疑点和研究惯量，不视作实机参数标定。固定基座展示的前馈没有用于训练。

## 当前范围

- 第一阶段：多高度站立、低速起停/行驶/转向、弱推扰；其它五阶段在此合同中尚未启用。
- 自由基座、真实PhysX接触；物理1000 Hz、策略100 Hz，solver迭代32/8。
- Actor230、critic92、6动作；包含实际伸缩坐标计算出的气簧压缩量/速度。
- 气簧每物理步施力，移动副effort上限1000 N；旋转关节与气簧分组，避免把推力截成100 N。
- 90–100%行程保留区沿用标注的三次曲线研究外推，越界单独记录；动力学研究不等于厂家高速工况验收。
- PPO沿用官方RSL5.5.1；每100更新保存checkpoint，最终导出ONNX并做ORT数值对照。

## 已完成本机检查

- 8 env × 8更新，从零初始化，3072 transitions，actor确实更新；最大闭合误差0.02774 mm。
- 从该checkpoint恢复2更新，累计10更新/3840 transitions，最大闭合误差0.06015 mm。
- 230→6 ONNX导出及batch验证通过，最大绝对差5.96e-8。
- 初始未训练策略存在非轮触地终止；不把链路检查称为站立能力验收。

本地证据：`reports/v5_train_pilot_01/`、`reports/v5_train_resume_check/`。

## 启动与进度

```bash
python scripts/launch_v5_remote.py --commit COMMIT_SHA \
  --num-envs 256 --output reports/v5_launch_new --execute
```

该入口将指定commit归档成独立快照，源码目录叶名为 `isaac_wheeled_rl_train`，启动专属tmux。
未传 `--execute` 只保存计划；`--updates` 用于带标签的有界工程检查。
正式第一阶段按491520000 transitions安排预算：256env对应40000更新，512env对应20000更新，
1024env对应10000更新，避免降低并行数时悄悄减少样本量。

run内：`startup.json`、`progress.json`、`model_*.pt`、`completion.json`、`policy.onnx(.json)`。
`progress.json`按正常完成的PPO更新计数，checkpoint记录累计训练样本量。
恢复需匹配同一合同/资产/控制输入身份，物理环境重新初始化，不声称轨迹逐位连续。

## 实时显示出口

远端启动默认加 `--publish-state`，最多4 Hz原子发布 `live_state.json`，仅取env0：
run/资产身份、真实19body世界位姿、xyzw约定、仿真时间、episode步数及墙钟时间。
此文件是实际post-step/自动reset之后的状态，网络采集与显示应保留age/STALE提示。
该出口为后续原生Kit镜像提供数据；它本身不是WebRTC视频服务。
