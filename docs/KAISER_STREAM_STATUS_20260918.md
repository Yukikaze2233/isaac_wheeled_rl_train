# Kaiser 原生GUI/WebRTC：2026-09-18核查

03:17–03:31完成独立检查。当前训练PID10560仍运行，更新从21155推进到21433；冻结清单21文件哈希一致。

## 结果

- 当前训练启动为 `--headless`，`LIVESTREAM=0`、`ENABLE_CAMERAS=0`、render关闭，未启用姿态publisher。
  当前进程没有可直接接入的原生画面/状态发布端口。仅设置新shell环境变量或端口转发不会改变运行中进程。
- WSL Vulkan仍报 `ERROR_INCOMPATIBLE_DRIVER`；CUDA PhysX可训练不等于该环境可渲染/串流。
- Windows原生runtime和streaming配置存在。新测TCP信令路由往返成功；原始UDP隧道对
  64/1200/1600/4096/60000字节报文全部往返、哈希一致。普通SSH `-L`不能单独承载UDP。
- Windows可用RAM三次为4.18/0.93/2.53GiB，低于已有单环境回放入口12GiB门槛。
  本次没有启动Kit或官方客户端，**未新验证WebRTC协商、实际视频或GUI输入**。
- 所有本次echo/tunnel探测进程退出，临时49100/47998端口与forward释放，原SSH连接和训练保留。

## 下一步

优先在宿主可用RAM恢复后，绑定一致checkpoint、源码、合同和地面资产，做Windows原生单环境90秒回放，
再用官方客户端确认画面/交互。此时看到的是独立回放，不是当前训练同一时刻的环境状态。

下一轮若要看真实训练状态，启动时接入低频姿态publisher并由Windows Kit渲染镜像，
或在有可用Vulkan的runtime上直接启动带renderer和streaming的训练。
镜像需携带run/asset身份、body顺序、xyzw、仿真时间与数据age；过期显示STALE。
Windows Kit接收桥目前尚未实现。

已有脚本与历史完整Kit/Cube串流证据在`isaac60`分支的`docs/KAISER_NATIVE_GUI.md`、
`scripts/windows_native/`。Cube历史成功不等于当前训练GUI已可连接。
本次完整命令、回执和清理记录保存在本地`reports/kaiser_stream_probe_20260918/REPORT.md`。
