# Kaiser V5 正式第一阶段：2026-09-18

更新：05:47在580次成功更新处正常暂停，checkpoint/ONNX已保存，28项产物已回收并逐文件校验。
当前任务已切换到[1024环境并行混合训练](V5_MIXED_RUNNING.md)，以下保留基础阶段启动记录。

**北京时间04:22启动，从零训练。**冻结源码commit `30db2b452c2f815a7d853f9baa838d4903389e8b`。
512环境、48步rollout、计划20000更新，约491520000 transitions，与1024环境×10000更新等样本预算。
物理1000Hz/策略100Hz、19刚体/18树轴/6闭环、10MPa非线性气簧；最长学习预算72小时。

04:25首次正式进度核对：PID329562已完成22次更新，约2600 transitions/s；
按此早期速度约52小时，后续以实测滑动窗口为准。旧Round4 PID10560同时仍在运行。
第一阶段当前为站立、变高、低速行驶/转向和弱推扰，完整课程的后续场景需阶段评估后启用。

## 精确位置和核查

- tmux：`v5-foundation-20260917T202156Z-b8a238`
- 远端根目录：`/home/kaiser/robot-rl-sim60/experiments/v5-foundation-20260917T202156Z-b8a238/`
- 日志：根目录的`train.log`；进度、checkpoint和实时位姿在`train/`。
- 本地原始回执：`reports/v5_kaiser_foundation_20260918/launch.json`。
- 可复用精简回执：[evidence/v5_foundation_launch_20260918.json](evidence/v5_foundation_launch_20260918.json)。

```bash
python scripts/check_chassis_remote.py docs/evidence/v5_foundation_launch_20260918.json
```

检查器读取当前进度、进程、内存、GPU和真实位姿发布，写入本地忽略目录，不以历史PID代替活性证明。
恢复同一任务时使用新run目录与经过身份校验的checkpoint；本次正式训练没有使用短测权重。

## 启动前验证

本机8环境从零8更新，再恢复2更新，230→6 ONNX数值验证通过。
Kaiser64环境4更新、512环境4更新均完成并导出验证成功：

| 规模 | 样本数 | 最大闭合误差 | ORT最大绝对差 |
|---|---:|---:|---:|
| 64 env | 12288 | 0.06564 mm | 8.94e-8 |
| 512 env | 98304 | 0.09156 mm | 5.96e-8 |

两次远端短测均无膝/气簧行程越界，未进入气簧曲线末端外推区。初始策略非轮触地频繁，
因此这些结果是工程与数值验证，不是站立技能验收。

## 已打通的实际显示链路

V5训练进程将env0的真实19刚体位姿原子发布到`live_state.json`，最多4Hz。
本机 `scripts/view_chassis_live.py` 通过持续SSH通道读取，检查资产身份、body顺序和xyzw单位四元数，
在原生Kit中只渲染这些位姿；**没有本地第二套物理或策略**。

窗口标题：`KAISER V5 LIVE TRAINING STATE | NATIVE KIT MIRROR`。
已实际接收超过140帧，验证错误0，显示LIVE，保存了原生viewport截图；观察到的source age约0.2–1 s。
该age使用两台机器的墙钟，是显示新鲜度参考，不是网络单向时延标定。
训练并行采样下单个env的模拟时间比墙钟慢，不能按画面墙钟推断实际指令速度。

```bash
OMNI_KIT_ACCEPT_EULA=YES python scripts/view_chassis_live.py \
  docs/evidence/v5_foundation_launch_20260918.json \
  --output reports/v5_live_view_new
```

实时窗口保留到用户关窗；超过2 s显示STALE。当前本机会话为`v5-kaiser-live-view`，
证据在`reports/v5_kaiser_live_view_01/`。断开此查看器不会终止远端训练。

## WebRTC视频层的状态

03:52再次独立验证了TCP49100和原始UDP47998通道：64/1200/1600/4096/60000字节5/5往返、哈希一致。
Windows可用RAM约5.69GiB，未达到已有原生回放入口12GiB门槛，因此本次未启动Windows Kit视频编码/客户端。
当前已展示的是上述**SSH位姿→本机Kit镜像**，不是把报文回环测试称为WebRTC视频成功。
原始探测回执见`reports/kaiser_stream_probe_20260918/FOLLOWUP_0353.md`。
