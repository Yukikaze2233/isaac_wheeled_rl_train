# 原生六轴探索与更新 KL 补充读取

这是新增只读 sidecar，复用字节未改的 `read_recovery_window_021.py` 及其020/019 CRC parser。它不增加训练 producer、不修改旧50/100回执、源码、合同或训练进程。

固定区间为本 attempt 的 actor11–110：task actor3086–3185、TB step30016–30115；父总消费29996与10个已消费 critic更新保持分开。严格读取6个 `DiagnosticUpdate/action_std_0..5`、`DiagnosticUpdate/kl_rollout_probe`、95个静态组的 `DiagnosticUpdate/<group>/kl`，另读 `Policy/mean_std` 作一致性参考。每个 tag 保留全部100个真实点及两半50统计；缺失为 pending/None，不能用 aggregate补成单轴或造零。

`LearningAudit.drain()` 已合并 `PolicyUpdateProbe.metrics()`，`train_chassis.py` 每次更新将结果写入原生 event。此前50/100 reader没有选这些tag，不能据此说TB不存在单轴std或KL。这一修正不更改既有质量读数。

policy顺序是LH、LK、RH、RK、LW、RW；control顺序是LH、LK、LW、RH、RK、RW。当前腿目标偏移为原始 action×0.25rad，轮目标速度为action×10rad/s；裁剪分别为±3与±9。报告中的std×缩放是**未裁剪的线性目标尺度**，轮缘再乘经过原生门绑定的0.06m半径。它不是实测腿速度、轮速度、车体速度、力矩或gyro噪声；裁剪、角度wrap、PD、通信、延迟和闭链被动力会改变实机／仿真响应。不能将腿位置std直接与2rad/s连续门比较。

KL是在每个静态组至多16个lane的同一最后rollout观测上，比较更新前后 raw policy distribution；它不是完整运动轨迹KL，也不说明连续1秒门的成功概率。group→global核对按该probe的lane数加权，不按全部环境人口加权。

root执行时需上传新helper，以及三份原冻结reader、first PPO和inventory（后三份字节不可改）。示例，路径由root的上传落点填写：

```sh
/home/kaiser/robot-rl-sim60/env/bin/python -B read_per_axis_std_021.py \
  --launch /home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/recovery_quality_launch_021.json \
  --first-ppo FIRST_PPO_VERIFICATION_021_JSON \
  --inventory FROZEN_SOURCE_INVENTORY_021_JSON \
  --reader021 READ_RECOVERY_WINDOW_021_PY \
  --reader020 READ_RECOVERY_WINDOW_020_PY \
  --reader019 READ_RECOVERY_WINDOW_019_PY \
  --output per_axis_std_021_001/report.json
```

这只验证当前worker的实际argv、PID/PPID/starttime连续性，不声称扫描了全部三层owner或排除了其他GPU进程。脚本不自行SSH、启动App/GPU/PPO或TensorBoard服务。没有读取到真实event前，不宣称已取得该100点结果或探索因果。
