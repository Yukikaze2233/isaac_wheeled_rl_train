# 当前 REC 接稳观测附件候选

只增加 `attach_holding.py` 附件，复用既有 source023 同封存诊断入口、`hold_joint_trace001.HoldingTrace` 与机械观测器的 `Hooks`。没有新增训练器、控制器、模型选择器或进程管理器。当前没有运行 App、GPU、原生场景或 PPO；`qualification=null`。

附件通过上下文暂时包裹原 `source023_runtime_observer` 工厂。工厂先调用原实现一次，再安装选中 lane 的被动 journal。原入口继续负责来源、合同、完整 seal、模型、RNG 和原 600 策略步/2400 物理步循环的认证。退出时只恢复自己拥有的工厂；发现他方覆盖时拒绝抹掉他方状态。

调用端仍使用原入口参数与生命周期。以下是未来调用端的接线形式，**本轮没有执行 `driver.main`**：

```python
manifest, binding = adapter.authenticate(manifest_path, manifest_sha)
driver = adapter.load_bound(manifest["driver"], binding[manifest["driver"]], "bound_source023_driver")
with adapter.attached(driver, manifest_path, manifest_sha):
    result = driver.main(original_same_seal_arguments)
```

`original_same_seal_arguments` 必须绑定真实的新 source023/v21 不可变封存及其 admission。附件不生成它，也不把旧 34000/REC6795 admission 当作当前。root 独立取得的 `immutable_update39000/` 数据和 CPU 核验属于输入准备；仅存在文件不代表原入口已经认证或执行新轨迹。

## 写入的数据与时钟

- 每个原 5 ms `FSM.update`：原 q/dq、重力、角速度、估计高度/support/body_clear；八项原门掩码；调用前后 phase、stable/peak/break、release、现有 lesson/profile 张量。函数仍调用一次、返回原对象。参考调用 k 的输入对应名义已完成物理端点 k−1，USB 反馈年龄仍未观测。
- 每个原 20 ms `begin_policy` 与策略步末：原 motion_hold、接管有效 mask 和整个物理区间的 `extras.policy_learning_mask`；episode reset 序号与出生身份分开，FSM 内部释放 reset 不冒充新 episode。
- 原 50 Hz `outcome`：原真实接触、undesired、姿态/角速度、高度、速度、平台与连续稳定/后续成功状态。它属于物理端点 4n；不能当作参考调用 4n 所观察的端点 4n−1 的真接触。
- 原 20 ms hold-cost producer 已保存的 mask/error/density：没有再次调用 reward producer；缺字段就记未观测。

固定八条 lane 来自真实冻结 v21 配置与原 `choose_scene_groups` 的 4096 分配：0、99、393、491、589、687、1373、1471。只选择原地形/行为行，不改变出生 lesson/profile；实际 L2 与正 dwell 覆盖仍取决于真实运行。启动时实际 group/kind 必须与独立 manifest 声明相同。

5 ms 只复制选中 lane 的原 device 张量，不读取全部 4096 lane 到 CPU。策略步结束后才序列化 `holding_intervals.jsonl.gz`，立即清空 journal 内存历史，保持事件上限。`holding_summary.json` 记录写入条数、调用次数、溢出及缺失范围。附件不安装完整机械 trace 的 PD/effort/closure hooks。

## 当前可用与欠缺

`dependency_readiness001.json` 独立重算了旧 `trace_manifest004.json` 的全部 25 个输入 SHA，包括 PT，全部实际匹配。原冻结 `/tmp` 源目录已经恢复存在；这不是本轮重建或复制的产物。此前 `hold_gate_trace004` 是简称，实际构件在 source023 的 `latest_seal_mechanical_trace001/`。

旧 004 的三个必需文件仍缺：`ROOT_CPU_mean004.receipt.json`、`root_trace_review004.json`、`lifecycle_manifest004.json`；六个旧 `/tmp` 原始诊断日志也缺失。附件没有伪造这些文件，旧生命周期入口不能据此声称就绪。新的当前 seal admission、该 seal 下完整 native 认证、原生轨迹和运行时开销尚待完成，当前占用 GPU 的 GUI 未触碰。

没有完整 5 ms 真接触或接触对明细、USB 反馈时间戳/年龄，也没有额外 simulator/contact getter。不能仅凭八门边际或 50 Hz outcome 推断每一次 1 秒联合接稳失败的原因。CPU 检查只证明被动接线与原函数语义，不能证明当前模型接稳或部署合格。
