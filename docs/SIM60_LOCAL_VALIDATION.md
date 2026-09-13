# Sim 6 本机训练验证（2026-09-12）

后续的64–512环境容量测量、GUI实际训练截图和地面过滤接触验证见 [容量与GUI记录](SIM60_LOCAL_CAPACITY.md)。

## 结论

**本机能运行 V40 GPU PPO 训练，并完成 TensorBoard / checkpoint / CPU ONNX 导出闭环。**
验证规模为 8 环境×3 次更新，以及 16 环境×2 次更新；这只是运行能力验证，未验证收敛、策略质量或不同仿真版本的数值等价性。

## 工作树与环境

- 工作树：`/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_train-60`
- 分支：`isaac60`；HEAD：`4ca58112c1ab23bea805a4d40ce707e61d244ef6`。开始时工作树干净，兼容修改留为未提交状态。
- 原 `isaac_wheeled_rl_train` 工作树为 `main` / `ee6b641926723a2b6f98ff31e59578f337353d0d`；原有和并行产生的未跟踪文件均保留。未切分支、提交、推送或访问服务器。
- Python：`/home/yukikaze/isaacsim60-venv/bin/python`，3.12.13。
- Lab 源码：`/home/yukikaze/Documents/workspace/robot_rl/.deployment_sources/IsaacLab-3.0.0-beta2`。
- Lab tag：`v3.0.0-beta2.patch1`；commit：`ffff603eafc6b74264a5261cc0183d6a65390d78`；官方 origin，tracked tree 干净。
- GPU：RTX 4060 Laptop，8188 MiB；驱动 610.57.04。系统内存约 30 GiB（标称 32GB）。

| 组件 | 实际版本 |
|---|---|
| isaacsim | 6.0.0.1 |
| isaaclab / assets / tasks / rl | 6.1.14 / 0.3.4 / 1.10.9 / 0.5.5 |
| isaaclab_physx / warp-lang | 1.1.3 / 1.13.0 |
| rsl-rl-lib | 5.5.1 |
| torch / CUDA wheel | 2.11.0+cu128 / CUDA 12.8 |
| torchvision | 0.26.0 |
| onnx / onnxruntime | 1.23.0rc1 / 1.30.0 |
| tensorboard | 2.21.0 |

直接使用用户现有 Sim 6 环境，没有安装依赖。最初 GPU 使用约 76 MiB；验证中用户另外启动了 Sim 6 GUI（PID 123023，命令为该 venv 的 `isaacsim` 入口），其 GPU 内存约 1135 MiB。GUI 保持运行，下面显存数字为**每秒采样的整卡峰值，包含 GUI/桌面**。

## 实测结果

日志根目录（相对本工作树）：`reports/local_sim60_20260912_0748/`。

| 运行 | 物理检查/更新 | transition | 训练阶段 | 整个子进程 | 整卡峰值 | 结果 |
|---|---|---:|---:|---:|---:|---|
| `physics07` | 2 env × 40 policy steps | 80 | — | 12.349 s | 3620 MiB | 40步完成，0提前终止 |
| `ppo01` → `train01` | 8 env × 48 steps × 3 PPO updates | 1152 | 6.793 s | 19.505 s | 3662 MiB | 训练/保存成功；旧导出器失败，保留原 receipt |
| `export01` | 使用 `train01/model_final.pt` | — | — | 2.061 s | — | 修复后独立 CPU 导出成功 |
| `ppo02` → `train02` | 16 env × 48 steps × 2 PPO updates | 1536 | 4.046 s | 17.452 s | 3660 MiB | 完整闭环成功，退出0 |

`train02` 的官方日志记录 381、399 transition/s（包含额外逐步审计开销）。两次更新不是稳定吞吐 benchmark，不能直接推算大规模长期训练性能。

最终验收证据：

- `ppo02/runtime_audit.json`：官方 `rsl_rl.runners.on_policy_runner.OnPolicyRunner` / `rsl_rl.algorithms.ppo.PPO` / `MLPModel`，actor 与物理均在 `cuda:0`；96个 vector steps 全部检查观测、动作、reward、done 和真实物理状态有限；actor 权重确实变化；训练时 `render_enabled=false`。
- `train02/completion.json`：`completed_updates=2`、`requested_iterations_completed=true`、`status=completed`、`export_status=verified`。
- `ppo02/artifact_audit.json`：68个 checkpoint/optimizer 张量全部有限；47个 TensorBoard scalar tags 全部有限，loss 的更新编号为0、1，`Termination/nonfinite` 为0。
- `ppo02/actual_policy_samples.pt`：32条真实仿真观测及最终 GPU actor 输出。逐条输入 `[1,125]` 的 CPU ONNX，输出 `[1,6]`；最大绝对误差 **7.450580596923828e-08**。
- `train02/export.stdout.log`：原导出器另外9组校验样本的最大误差 `7.152557373046875e-07`。
- `train02/model_final.pt`、`policy.onnx`、`policy.onnx.json` 和 TensorBoard event 文件均已保存。

### 关节与观测契约

资产和 `contracts/own_v40_v2.json` 没有修改，继续使用统一 `locomotion`：actor125 / critic29 / action6，200Hz物理、100Hz策略，MLP `[256,128,64]` / ELU，无观测归一化。

在 reset 后、40步后和最终16环境训练后读取**实际 PhysX solver**：四个髋/轮 continuous DOF 为精确 `[-FLT_MAX,+FLT_MAX]`。两膝为契约所指定的机械内角35–80°对应的 URDF 坐标：

- `L_joint2`：`[-0.1267274171, 0.6586707830]` rad。
- `R_jonit2`：`[-0.6119707227, 0.1734274179]` rad。

每个克隆都检查，未放宽容差或用缓存 soft limits 替代 solver 回读。本次没有执行多圈运动实验。七刚体 contact history / net force 形状及实际非零接触读数已验证；USD 六对相邻体过滤关系检查保留。

## 兼容修复

1. URDF 导入配置改为 `collision_type="Convex Hull"`；USD stage 使用 Lab 3 的 `sim.stage`。
2. 显式使用 Warp `ProxyArray.torch`，新 `*_index` 写入 API 和 int32 joint indices；XYZW 在几何公式边界正确处理，评估 snapshot 仍输出 WXYZ。
3. 新 `V40ContactSensor` 只替换嵌套 URDF 刚体路径的解析/绑定，复用官方 contact buffers、kernels 和更新逻辑。为每个真实刚体激活 reporter，并严格验证 PhysX 返回的环境/刚体顺序。
4. 复用 Lab 官方 `handle_deprecated_rsl_rl_cfg`，把原配置转为 RSL 5 的独立 actor/critic；观测组显式绑定 actor→policy、critic→critic。
5. checkpoint 导出增加显式 provenance 约束的 RSL 5 split MLP 格式；检查两张网络的完整键集合、shape、dtype、有限值和归一化状态。只在内存中转换键名，不改写 checkpoint。CPU 导出子进程隔离仍保留。
6. Sim 6 `SimulationApp.close(exit_code=...)` 保留真实失败退出码，并在 Kit 快速退出之前打印异常，避免“退出0但没有执行物理步”的假成功。
7. 入口重新启用该 Sim 6 栈的精确版本 gate，并写入实际 runtime / Lab source provenance；本机 launcher 指向 Sim 6 + V2 locomotion。

生产修改：`scripts/{train_v40,check_v40_env}.py`、`src/wheeled_algo/{v40_export,v40_job}.py`、`src/wheeled_tasks/direct/v40_serial/{env,contact_sensor}.py`、`src/wheeled_world/assets/v40.py`、`v40_train_local.sh`。

## 复现

在上述 `isaac60` 工作树中运行；`--run-dir` 必须使用新目录：

```bash
timeout --signal=TERM --kill-after=30s 600s env \
  OMNI_KIT_ACCEPT_EULA=YES \
  TMPDIR=/home/yukikaze/.cache/kit-tmp \
  LD_LIBRARY_PATH=/home/yukikaze/.local/lib/compat \
  /home/yukikaze/isaacsim60-venv/bin/python scripts/train_v40.py \
  --research --headless --contract contracts/own_v40_v2.json \
  --stage locomotion --device cuda:0 --seed 40 \
  --num-envs 16 --max-iterations 2 --max-runtime-seconds 180 \
  --run-dir reports/local_sim60_recheck/train \
  --usd-cache-dir reports/local_sim60_recheck/usd_cache
```

此次实际运行使用独立 tmux session `v40-local-sim60-20260912`，每一阶段退出后 session 自动结束。精确 argv / PID / elapsed / exit code 见各阶段 `summary.json`，stdout/stderr 合并在 `output.log`，GPU 采样在 `gpu.jsonl`。

本次本地审计脚本：`/tmp/opencode/sim60_local_bounded.py`（600秒外层预算）、`sim60_audited_train.py`（调用生产入口，只包装 env.step 做断言，不替换 PPO）、`sim60_audit_artifacts.py`（CPU 真实观测/ONNX/TensorBoard 审计）。

## 回归与已知范围

以下测试在 Sim 6 venv、`CUDA_VISIBLE_DEVICES=''` 下 **300 passed**；JUnit 为日志根目录的 `cpu_tests.xml`：

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' /home/yukikaze/isaacsim60-venv/bin/python -m pytest \
  tests/v40/test_core.py tests/v40/test_joint_import_limits.py tests/v40/test_launch.py \
  tests/v40/test_round2.py tests/v40/test_unified_commands.py \
  tests/v40/test_sim60_compat.py tests/v40/test_export.py -q
```

`bash -n v40_train_local.sh`、`git diff --check` 通过。覆盖包括真实 RSL 5 MLP 导出、拒绝缺失/混合/NaN/错误shape权重、原限位拒绝测试、XYZW几何公式、噪声/历史/命令逻辑、native checkpoint 恢复及错误退出码。

另外执行的既有 `test_assets.py::test_mujoco_static_parity_without_stepping` 在本 venv 的 **MuJoCo 3.8.0** 下失败：`minimum_nonadjacent_geom_distance.distance_m` 返回0，旧断言要求大于0.006；其工具代码、资产和测试未修改，根因未在此次训练兼容任务中定位。不能把上述300项通过表述为整个仓库全部CI通过。其余未列出的旧运行/录制/远程管理入口也未在 Sim 6 上逐一验证。
