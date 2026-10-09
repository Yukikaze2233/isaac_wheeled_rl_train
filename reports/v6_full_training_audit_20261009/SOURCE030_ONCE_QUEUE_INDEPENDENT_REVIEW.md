# source030 一次正常边界队列：独立审查

结论：协调器 SHA `0478bfe38c673e7fdb9ad1a5a6fb6151c76848e2a4182714b8910c8c357eb339` 未见遗留静态阻断；交 root 冻结和决定实际入队。本审查未执行远端命令、信号、native controller、准入、launch、PPO 或重复 CPU 测试。

固定 queue flock 贯穿一次流程，独占 intent 和旧产品检查阻止失败后重排。等待交给原冻结 controller 的604800s参数；原控制器在最后20–100 actor窗口才暂停认证supervisor，worker正常完成。协调器没有自己的signal/kill操作，Popen.wait没有超时杀child逻辑；它不占原账户锁等待worker。

只有真实成功controller receipt（worker signals0、无后续dispatch、原生0PPO闭合）才将其实际SHA交给既有helper。随后native boundary-prepare→认证旧交付保留→原helper launch，未来parent和source alias不在队列中预填。提交后仍为first_terrain_actor_pending，不能称新课程已学习或已通过。

审查指出的来源缺口已修：不只验证controller/publisher两个入口，而是在controller启动前复用SHA已认证helper，对4341文件完整归档/权限/解包字节核对；等待后和launch前重做。非controller依赖和权限变更的CPU负例在execute之前拒绝。冻结stage/archive/runtime/helper及解释器pin保持。

旧delivery三个硬链接已按**实际源码**核对：e5与81ad的remote_job.finalize对delivery.tar.gz和delivery.json使用tmp.replace；archiver_status.json由main导入的chassis_batch_export.write_json写入，同样.json.tmp后replace。因此原生下一轮更新路径不会改旧inode。代码按prepare返回的真实路径保留，README路径已纠正；只能在原子发布和同文件系统前提下使用。

作者16项CPU通过/14.482s：真实helper归档和权限读取、native原子publisher、临时子进程flock、原子replace后硬链接保留；controller/prepare/launch用synthetic receipts和execute spies，未执行生产链。本人重算源码、测试、文档、日志SHA与result001一致。

SIGKILL、宿主丢失或父进程取消时controller可能继续，不能声称finally覆盖全部故障；必须查持久intent/PID和实际产物，禁止自动重排。精确来源SHA、检查项与限制见`source030_queue_independent_review001.json`。实际Kaiser入队/正常终局/terrain首actor尚须root另取证。
