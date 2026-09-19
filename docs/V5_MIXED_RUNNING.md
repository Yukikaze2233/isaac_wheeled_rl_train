# V5 并行混合训练：历史运行记录

**状态更新：2026-09-19 22:46（UTC+8）已正常停止，19179更新；217项产物已回收校验。**
[停止、回收和固定场景评测](V5_MIXED_RECOVERY_AND_EVALUATION_20260919.md)记录最终结果。
下文为启动时的参数与操作记录，缓存live状态不表示进程仍在运行。

**2026-09-18 06:02（UTC+8）正式启动**，源码冻结在`c65a67781f27ee67c1f187c6dd99c351fabcaede`。
1024辆车同时训练五类任务：站立256、平移256、旋转204、上阶/跳跃153、下阶/落地155。
地形包括425个基础平地实例及材质变化、坡面、起伏、平台、低坎、单阶、连续低阶和落差。

父模型是V5基础阶段580更新checkpoint，SHA256：
`3b50ef7d006f5ffd3c87827d8079210b25cb705b09bf11e70dda807283abb8e6`。
使用权重迁移和新优化器；不是旧七刚体权重，也不是把不同任务当作精确resume。

总预算100000更新、约49.15亿样本。**72小时是本次运行块上限，不代表72小时跑满全部预算**。
早期1024容量实测约4430 transitions/s，比512基础检查约2700更高；正式首次核查总GPU利用率92%，
WSL仍有约5.46GiB可用RAM。旧Round4 PID10560保留继续运行。

## 进度、实时查看和恢复

精简回执：`docs/evidence/v5_mixed_launch_20260918.json`。
远端根目录：`/home/kaiser/robot-rl-sim60/experiments/v5-mixed-20260917T220232Z-ee7970/`。
tmux同名；初次确认worker PID378126，已完成16更新。当前状态以查询为准。

```bash
python scripts/check_chassis_remote.py docs/evidence/v5_mixed_launch_20260918.json

OMNI_KIT_ACCEPT_EULA=YES python scripts/view_chassis_live.py \
  docs/evidence/v5_mixed_launch_20260918.json --output reports/v5_mixed_view_new
```

查看器下拉框选择五个任务组的代表实例。训练并行数与渲染实例数分开，显示一辆不代表只训练一辆。
画面来自远端真实PhysX位姿，本机Kit只渲染；这是SSH状态镜像，尚不是WebRTC视频。
上阶组默认代表是低坎，组内还有单阶、连续阶和跳跃，完整分布可查`startup.json`。

## 力矩判断与已发现问题

每个物理步采集裁剪后的实际执行器输出，每10更新刷新`torque_monitor.json`并追加`torque_history.jsonl`，
按五组区分六个电机的峰值/RMS/转速/正负机械功率/相对即时曲线饱和比例；气簧N、速度与行程单列。

容量测试中，腿电机峰值约24Nm、轮子最大约3.83Nm，没有普遍持续饱和；但大量样本停在低位，
气簧进入末端保留区，高度误差约6–7cm。**因此不能认定“力矩没超就合理”**。
正式合同增加连续高度跟踪、恢复阶段高度目标和统一机械工作余量代价，避免靠硬止挡长期支撑。
新代价是否有效仍需观察训练趋势和独立验收；模型惯量、源质量不对称及电机连续额定值仍有未标定项。

## 自动回收已经接通

- 远端训练子进程退出后自动生成`delivery.tar.gz`及`delivery.json`，包含final/周期checkpoint、ONNX、
  配置、源码快照、TensorBoard、力矩历史、训练日志和完成记录。
- 本机`v5-mixed-recovery`独立tmux等待交付，下载并逐文件验证SHA256，目标为
  `reports/v5_mixed_recovered_20260918/`。
- 本机关机不影响远端训练/打包；本机重新在线后用相同回执/目标目录重启watcher。

```bash
python scripts/watch_chassis_artifacts.py docs/evidence/v5_mixed_launch_20260918.json \
  --output reports/v5_mixed_recovered_20260918
```

此前基础阶段已经实际回收28项产物并验证通过，保存在`reports/v5_foundation_recovered_20260918/`；
训练状态保留为`stopped`，没有将正常暂停或字节回收成功改称为能力验收通过。
