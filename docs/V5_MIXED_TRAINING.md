# V5 多车、多场景同时训练

`contracts/v5_mixed_v1.json` 将同一策略的环境分为五组，参考华南虎的混合地形并行组织，
保留自有机构、10 MPa气簧及上层任务指令。不是等基础阶段跑完才生成其它场地。

| 组 | 比例 | 场景与命令 |
|---|---:|---|
| 站立/变高 | 25% | 平地、多材质、坡面；零平移/零旋转，高度独立重采样 |
| 平移/起停 | 25% | 平地、材质、起伏、低坎；正反向速度 |
| 原地旋转 | 20% | 平地、坡面、平台顶部；双向yaw，非平地先限低速 |
| 上阶/跳跃 | 15% | 低坎滚越、单阶、连续低阶、平地跳跃 |
| 下阶/落地 | 15% | 下台阶、落地吸能、稳定恢复 |

1024环境对应256/256/204/153/155辆；至少40%的实例是基础平地。
地形几何以25/50/75/100%四档同时存在；速度从0.5m/s、yaw1rad/s逐渐扩至阶段上限。
课程按成功更新后的训练样本量推进，不用物理tick冒充PPO更新。
这种实现参考了地形混合与课程思想，不宣称复刻华南虎的所有地形或成功率自适应晋级逻辑。

## 保留已有权重

基础训练先正常停止并保存。混合任务使用显式 `--transfer`：
验证资产、动作、观测布局和控制参数一致后，加载actor/critic权重，使用新优化器及新计数。
它不是把不同合同伪装成精确resume。

```bash
python scripts/launch_v5_remote.py --commit COMMIT \
  --contract contracts/v5_mixed_v1.json --stage mixed --num-envs 1024 \
  --transfer /absolute/remote/foundation/train/model_final.pt \
  --max-runtime-seconds 259200 --output reports/v5_mixed_launch_new --execute
```

总课程预算按1024env×48步×100000更新定义；72小时为一次运行块的上限，未跑满时保存并继续恢复。
并行数通过目标服务器实际容量测试确定，不把显存空闲或GPU利用率单一数值当作吞吐保证。

## 力矩与气簧监控

`torque_monitor.json` 每10次更新刷新，结束时保存最终统计，按五组分别记录：
每次累计统计同时追加到`torque_history.jsonl`，可回溯变化而非只留下最后一次结果。

- 显式执行器裁剪后、实际送入仿真的六电机力矩：峰值、RMS。
- 原始电机请求相对即时力矩—转速曲线的95%饱和比例。
- 关节转速峰值、正/负机械功率均值。
- 气簧轴向力峰值（N）、伸缩速度、压缩行程范围。

统计以物理步采样；电机Nm和气簧N分开，约束反力也不冒充电机力矩。
主动腿40Nm、轮子约3.84Nm及其曲线仍是研究参考，不能当成已核实的连续额定值。
尤其静止大扭矩时机械功率可以接近零，但电机仍可能发热；必须结合RMS、持续时间及未来电流/温度数据。
长期饱和需同时检查跟踪误差、姿态、源质量/惯量、力臂与气簧基准，不能只调大限幅掩盖原因。

基础权重在580更新停止时，平均回合已约16.97 s，但高度MAE约66.7 mm，不能称为变高达标。
新合同增加支撑/恢复阶段的连续高度L1代价（起始−10/m/s reward rate），并让RECOVERY阶段也跟踪高度，
避免只靠窄高斯高度奖励在远目标处缺少区分信号。腾空阶段仍不使用贴地高度保持。

## 实时查看

训练端最多4Hz发布五组各一个代表环境；查看器的下拉框可选 `stand / translate / rotate / step_up / step_down`。
画面显示一个代表实例，不代表训练只开一辆车。每个代表的真实碰撞场地描述也随状态发送。
这是SSH位姿流到本机原生Kit的镜像，不是WebRTC视频服务。

## 训练结束自动回收

远端任务由 `scripts/chassis_remote_job.py` 管理：训练子进程退出后，打包final及周期checkpoint、ONNX、
配置、源码快照、TensorBoard、日志和完成记录，生成`delivery.tar.gz`与逐文件SHA256清单`delivery.json`。
即使本机关机，远端训练与打包仍由tmux继续运行。

本机独立watcher等到清单就绪后下载，校验压缩包和每个文件：

```bash
python scripts/watch_chassis_artifacts.py reports/v5_mixed_launch_new/launch.json \
  --output reports/current/v5_mixed_recovered
```

`recovery.json`表示字节校验完成，保留训练的completed/stopped/failed状态，不把回收成功等同于训练成功。
本机关闭或SSH认证失效时，重新连接后用相同receipt/output重启watcher即可。
