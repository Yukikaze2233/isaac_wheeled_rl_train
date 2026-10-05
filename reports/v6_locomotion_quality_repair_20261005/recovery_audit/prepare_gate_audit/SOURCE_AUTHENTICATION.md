# PREPARE 单行物理 A/B 的源码认证入口

现有原始入口 `core_sources(ROOT)` 只接受 12 个 core 文件逐字节等于 `b4ce80964d47c0127e0b3d9929ad35a620bbf204`。直接对当前恢复类做 patch、monkeypatch 或替换 `sys.path` 都不是一个已认证的物理候选。

本轮仅扩展 `scripts/recovery_prefix_support.py` 与 `scripts/probe_recovery_physical_prefix.py` 的工程认证入口。**没有修改恢复 core、PPO 或合同，没有开始新的物理运行。**默认无 manifest 的旧入口仍要求完整 b4 core，不能接受候选。

## 两层冻结，两个完整源码根

1. 先冻结共同 harness **H**：包含本次两个认证脚本和测试；12 core 文件都必须等于 b4。H 可保留报告文件，但不是从现场 dirty 树运行物理。
2. 候选 **C** 从 H 的独立 checkout 构造，仅应用 [candidate_prepare_slew.diff](candidate_prepare_slew.diff)，提交该一行。H→C 的**整个 Git 树差异**必须只含 `src/wheeled_tasks/chassis/recovery.py`；候选文件必须逐字节等于 b4 文件的唯一固定替换。不能顺带改轮增益、速度门、超时、合同、任何其他文件。
3. A 使用完整 clean Git checkout H，B 使用完整 clean Git checkout C；从各自绝对 `scripts/probe_recovery_physical_prefix.py` 路径执行。脚本的 `ROOT/sys.path` 自然落在自己的完整 source root，禁止把候选脚本搭在当前正在训练的 source 上。
4. manifest、物理输出、模型放在源码 checkout **之外**。从 `/tmp` 启动绝对脚本，避免 Kit 或输出文件污染 clean source。用户当前 dirty 主树、正在训练 v14 的 frozen source、原始 002 都不动。

只有 H/C 已冻结后再生成 manifest；不能把拟定 SHA 或现场内容推断成已冻结版本。

## 外部 manifest 与明确身份

[freeze_source_identity.py](freeze_source_identity.py) 是单用途 CPU helper，不更改源码、不启动 AppLauncher/GPU/训练。必须显式传入 H；当前 source Git HEAD 是 A 的 H 或 B 的 C，不能靠父分支名称猜测。

```bash
python -B /absolute/repo/reports/v6_locomotion_quality_repair_20261005/recovery_audit/prepare_gate_audit/freeze_source_identity.py \
  --source-root /absolute/clean-source-A \
  --harness-revision <H完整40位SHA> --mode baseline \
  --output /absolute/external-manifests/baseline_source_identity.json

python -B /absolute/repo/reports/v6_locomotion_quality_repair_20261005/recovery_audit/prepare_gate_audit/freeze_source_identity.py \
  --source-root /absolute/clean-source-B \
  --harness-revision <同一H完整40位SHA> --mode prepare_slew_1_candidate \
  --output /absolute/external-manifests/candidate_source_identity.json
```

helper 输出认证回执及 manifest SHA，失败即删除失败 manifest，已有路径拒绝覆盖。冻结后的 SHA 应写进部署回执并作为下一条命令的独立输入，不让运行器根据当前内容自行接受新的 SHA。

manifest 的唯一 schema 为 `recovery-prefix-source-identity-v1`，字段集合固定：

| 字段 | 必须证明的内容 |
|---|---|
| `mode`, `source_root` | baseline/candidate 明确分开；实际源码根与 Git toplevel 相同 |
| `base_core_revision` | b4 全 SHA，不能改基线 |
| `harness_revision`, `source_revision` | 固定 H/C 全 SHA；实际 Git HEAD 必须相同且 checkout clean |
| `base_core_files_sha256`, `core_files_sha256` | 恰好原 12 core SHA；共同 H 均为 b4，候选只允许该一行推导出的 recovery SHA |
| `runtime_sources_sha256` | 原生 `sources()` 全部实际 Python 源码 SHA，与 source revision Git 对象逐文件一致，拒绝增加/少报/现场 overlay |
| `candidate_change`, `candidate_diff_sha256` | baseline=null 且 H=C；candidate=固定 file/old/new，H→C 的完整 Git diff SHA 完全对应 |
| `engineering_expectations_sha256` | 固定原生四姿态投影 SHA `a770d28ab43904295bed212b098ae5b51fb94258b68d40ce3b9330a4ae2273ae` |

基线 report 中 `revision=H`、`base_revision=b4`；候选 report 中 **`revision=C`**、`base_revision=b4`、`harness_revision=H` 和固定差异完整保留。不能把候选标成“b4 原样”。

## CPU 预检和物理入口

两侧分别先使用同一旧 flat PT `7e3644f...` 和自身源码根中的 V8 recovery worker 做 CPU 预检：

```bash
cd /tmp
python -B /absolute/clean-source-A/scripts/probe_recovery_physical_prefix.py \
  --contract /absolute/clean-source-A/contracts/v6_task_campaign_recovery_worker_v8.json \
  --checkpoint /absolute/original/models/v6_model_delivery_20261003T2235/recommended_flat/model_final.pt \
  --source-identity /absolute/external-manifests/baseline_source_identity.json \
  --source-identity-sha256 <A生成回执的完整64位SHA> \
  --output /absolute/external-reports/A_preflight --preflight-only
```

B 同式换 source、manifest、外部 output 和对应 SHA；两侧不能混用。CPU 预检仍认证原始 V8 worker/plan/design、严格 PT seal/infos/ABI，并独立重新物化四姿态 `engineering_expectations.json`，实际 SHA 必须等于上述 002 原始值。两侧除绝对输出路径等非行为 provenance 外，出生/种子/有效合同和设计应一致。

物理运行使用已经审过的同一命令，分别去掉 `--preflight-only`，改用新的不可覆盖输出目录；由 root 单独执行。保留 4 个第一出生、seed 290929、所有原始判定、8 s 脚本 budget、1 s 被动释放、40 tick BLEND、后续任务以及原有 no-PPO 记账。不得从事件快照声称恢复 PhysX 内部状态。物理结果仍是工程诊断，不自动取得 hardware 或训练 qualification。

认证在预检、原生 scene import 前和 finally 收尾再次检查。来源变化、Git 身份漂移或源码增加/减少都会留下失败状态；CPU preflight 的最后认证失败也必须返回失败退出码，不能靠早期成功 return 掩盖。

## 已完成测试，尚未物理运行

**37 项 CPU 测试通过**：原有 23 项 prefix 测试与新增 14 项身份测试。新增测试在 disposable Git fixture 中验证：baseline/candidate 正常路径；无 manifest 拒绝候选；错误/遗漏外部 SHA；额外 core 改动；同 recovery 文件任意额外 diff；Git HEAD、源码根、候选 diff、独立设计的错误绑定；dirty/untracked overlay；实际 runtime SHA 不符；原四姿态设计 SHA；收尾身份失败不能返回成功。测试不加载 Isaac 或启动 GPU。

```bash
/home/yukikaze/isaacsim60-venv/bin/python -m pytest -q \
  tests/test_recovery_prefix_source_identity.py tests/test_recovery_physical_prefix_probe.py
```

H/C 未在此报告中填入假定值，manifest 生成和物理 A/B 仍由 root 后续冻结执行。此次认证增强只回答“究竟跑了哪份 source”，不回答“1 rad/s 是否能修复 PREPARE”。
