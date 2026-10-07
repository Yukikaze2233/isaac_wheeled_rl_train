# 本机过时 reports 清理回执

完成时间：2026-10-08 03:42:26 UTC+8。用户已授权清理两项目的过时测试残留。本次只执行本机精确路径清理，远端操作与 PPO 动作均为 0。

| 操作 | 实际数量 | 回收的文件分配字节 |
|---|---:|---:|
| 内容 SHA 相同、mode/uid/gid/device 相同的旧副本原子替换为 hardlink | 469 个副本 | 3,609,571,328 |
| 已由完整 MP4 保留的历史中间渲染 PNG | 5,065 帧、25 组 | 1,801,240,576 |
| 无在用引用的测试浏览器 profile／字体与 Python 缓存 | 5 个目录 | 14,024,704 |
| 合计 | | **5,424,836,608（5.42 GB / 5.05 GiB）** |

60 个权限／属主／设备属性不一致的重复组原样跳过；未修改权限。所有重复副本原有路径与内容 SHA 均保留，只有已明确记录的副本 inode 改为相同内容的 canonical inode。历史报告应继续按不可变产物使用，避免以后向共享 inode 原地写入。

25 组历史 `viewport_frames` 现在仅保留各自首帧和末帧 PNG，共 50 张；完整 MP4、帧时间 `video_capture.json`、`report.json`、`runtime.json`、`traces_000.npz`、source snapshot 均保留。旧录像编码明确从 0 号图开始，输出 `round(seconds * FPS)` 帧，而 capture 包含一个额外终点；本次核对 MP4 的帧数、25 fps 和时长后，仅删除中间已编码帧，额外终点原图保留。两组缺视频或 capture 元数据的目录未清理。早期 README 中“原始 viewport PNG”描述对应清理前状态，当前以此回执为准。

模型、ONNX、bag、机械 USD/STL/URDF 等资产、当前 source023/latest100/current launch 与 native004 原始证据均在清理范围之外。所有保留视频、两端 PNG、capture 以及 hardlink 内容已终验 SHA；两个项目的 `git diff --check` 和 `git diff --cached --check` 均 exit 0。新训练版本、合同和预算未修改。

文件系统可用空间增量为 5,317,402,624 字节，该量同时受其他本机写入影响；回收量采用上述已消失的独立 inode 与 PNG/cache 文件分配字节计算。

- [实际清理结果](./reports_cleanup_20261008_safe003.result.json)，SHA `aebc887d395b4f467b3e2cb174016f6998409651f7ae12426f2b55001c6faa62`。
- [逐动作 intent / applied 日志](./reports_cleanup_20261008_safe003.actions.jsonl)，SHA `51a8f3bcd3ffd6e3b222cab9a790fce509ecf19228753f1697fa560a6e72f8bf`。
- [精确副本清单](./reports_cleanup_20261008_safe003.dedup.inventory.json)。
- [视频／中间帧与缓存清单](./reports_cleanup_20261008_safe003.frames.inventory.json)。
- [唯一执行脚本](./apply_reports_cleanup_20261008_safe003.py)，SHA `07649d1a11a1b8c0bff19e84bc2f7e014c3cc861ece58972adfa91f7d027dfd7`；带 once-only 回执检查，请勿重复执行。

旧 `apply_report_dedup_20261008.py` 在 mode 全预检阶段退出 1，未产生 result 或进行 mutation；原失败脚本与 inventory 留存，本次使用独立 safe003 执行。
