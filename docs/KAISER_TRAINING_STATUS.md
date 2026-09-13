# Kaiser 训练检查：2026-09-13

## 14:32–14:37 北京时间的实际状态

SSH 已重新连接并检查 WSL 进程、tmux、训练目录、完成回执及 Windows 原生部署。**这个检查时点没有正在执行的 V40 训练，没有正式 Round3 产物。** 后续新启动不能反向改变此记录。

| 已有任务 | 实际完成 | 含义 |
|---|---|---|
| `v40-live-snapshot03/v40-live-20260912T234215-100466/train` | 2 env，3/3 PPO 更新，ONNX 导出校验通过，9月12日23:42:37完成 | 实时显示工程短测，不是 Round3；`policy_quality_verified=false` |
| `round2-height-20260912T155346Z` | 第二轮最终 ONNX 的五高度评估均退出0，9月13日00:14:43完成 | 每高度累计60秒，含3次timeout；没有新学习更新 |
| Windows 原生 Sim / Lab | 安装完成，原生GUI曾在当前Linux收到视频并验证输入 | Cube场景；不是原生Windows V40/PPO训练 |

WSL训练根目录：`/home/kaiser/robot-rl-sim60`。扫描排除了SDK、缓存、下载和资产目录；Windows结果范围及原始GUI证据见[原生GUI记录](KAISER_NATIVE_GUI.md)。没有将GPU利用率或显存占用当作训练进度。

## 已拉回的完成证据

- [3-update smoke原始完成回执](evidence/kaiser-smoke-completion-20260912.json)
- [五高度评估原始批回执](evidence/kaiser-height-batch-completion-20260913.json)
- [五高度结果与口径](ROUND2_HEIGHT_EVALUATION.md)
- [第二轮最终结果审计](ROUND2_FINAL_AUDIT.md)
- [摩擦、滑移与高速物理预算审计](V40_FRICTION_AUDIT.md)

五高度都存在零速漂移；0.28 m有持续非轮净力。当前数据不含足够的轮地接触点/轮速遥测，不能把漂移直接认定为打滑。现有实际资产绑定和源码默认链推导的名义有效静/动摩擦均约0.5，合成模式为average；不是直接读取当时solver材质表的结果。

## 代码发布范围

Sim6/Lab3兼容、RSL5导出、真实诊断与终止分离、headless回放、独立实时几何显示及Windows原生GUI部署工具。外部`V40_USD_SEED`尚无与canonical资产和引用依赖的哈希绑定，当前入口和资产工厂均拒绝该实验路径，防止替换物理模型后仍记录原资产身份。

此检查不代表第三轮已启动或已完成。第三轮必须有独立合同/任务快照、启动命令、进程与首个checkpoint证据；最终回传仍需独立效果评估。
