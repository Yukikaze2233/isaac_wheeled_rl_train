# V4.0 FrameStack PPO：可信、拒绝式 ONNX 导出

这是独立的新导出路径，不修补/调用旧 `scripts/export_onnx.py`，不使用未定义的 `actor_dims_from_ckpt`。仅依赖 Torch / NumPy / ONNX / ONNXRuntime，**不导入 Isaac 或本机 rsl_rl，不训练、不做随机策略物理仿真**。

导出数值一致不代表策略已学会站立、运动或可安全部署；合成测试 checkpoint 从未训练。

Round4 最终模型的显式 FP64 内部计算选项及保留失败证据的恢复流程见
[ROUND4_EXPORT_RECOVERY.md](ROUND4_EXPORT_RECOVERY.md)。默认 FP32 验收不变。

## 1. 冻结接口

- `contract_id = own-v40-jointspace-h5-v1`，manifest `schema_version = 1`。
- Actor：物理观测 **25 × 5 = 125**；ONNX 输入 `obs_history: float32[1,125]`。
- Critic：仅训练用 **29 = 当前物理25 + privileged 基座线速度3 + 高度1**。校验 critic 权重，但不导出 critic。
- 输出：`actions: float32[1,6]`，**原始确定性 actor mean**，无噪声采样、末层激活、动作裁剪或电机缩放。
- 动作序：`[L1,L2,L3,R1,R_jonit2,R3]`；对应真实关节 `[L_joint1,L_joint2,L_joint3,R_joint1,R_jonit2,R_joint3]`。保留资产中的 `R_jonit2` 拼写，不按索引/正则重新排序。
- 第一版项目配置：stock `ActorCritic`，actor/critic hidden dims 都是 `[256,128,64]`，ELU，所有 empirical observation normalization 关闭。架构字段仍必须显式记录，导出器不自动补默认值。测试可显式声明较小 MLP。
- 固定 batch=1、opset=17、`dynamo=False`、无 dynamic axes、无外部权重文件。

## 2. 已核对 RSL-RL **v3.0.1** 官方源码

只读获取官方 tag 源码，未安装/执行 RSL 3.0.1，也未以本机 2.3.3 冒充目标版本。

| 官方文件（`leggedrobotics/rsl_rl` tag `v3.0.1`） | 本次读取内容 SHA-256 |
|---|---|
| [modules/actor_critic.py](https://raw.githubusercontent.com/leggedrobotics/rsl_rl/v3.0.1/rsl_rl/modules/actor_critic.py) | `20c6396d32802112dc217ba51add5fe7f7953518a0c7cdb4b7da51f77dc30b6e` |
| [networks/mlp.py](https://raw.githubusercontent.com/leggedrobotics/rsl_rl/v3.0.1/rsl_rl/networks/mlp.py) | `ea52cb5b27e6d986731ac657bcc37f8b74a1e5630e8a8fb323986ed2b6c01b23` |
| [runners/on_policy_runner.py](https://raw.githubusercontent.com/leggedrobotics/rsl_rl/v3.0.1/rsl_rl/runners/on_policy_runner.py) | `e51e4dd49664ccc8743a16b71d2971d392f40a6988136f10fe8c5f03ab6a83b5` |
| [utils/utils.py](https://raw.githubusercontent.com/leggedrobotics/rsl_rl/v3.0.1/rsl_rl/utils/utils.py) | `8256eaa30982b88f6383dfa5f35467369bc52503c2796a07568961bbc5070a74` |

确认：

1. `OnPolicyRunner.save()` 的顶层是 `model_state_dict`, `optimizer_state_dict`, `iter`, `infos`；`model_state_dict = self.alg.policy.state_dict()`。`infos` 默认 `None`，所以父训练入口**必须为所有自动/中间/final 保存注入 metadata**，否则本导出器拒绝。
2. `ActorCritic.actor` 是 `MLP`，后者继承 `nn.Sequential` 并用数值字符串注册层。因此三 hidden 层对应 `actor.0/2/4/6.weight/bias`；critic 同样。Actor 最后是 Linear，无末层非线性。
3. Policy 构造 API 是 `obs, obs_groups, num_actions`，不是 2.x 的两个观测维度参数。项目必须给正确 groups（actor=125，critic=29），导出器不承担训练 API 迁移。
4. v3.0.1 归一化位于 policy：`actor_obs_normalization`, `critic_obs_normalization`；关闭时 normalizer 是无 state 的 `Identity`。**父训练配置显式关闭两者**；仅旧 runner `empirical_normalization=False` 不是替代设置。
5. 官方 hidden 默认是 `[256,256,256]`，不是项目选定的 `[256,128,64]`；父训练配置必须显式传入。
6. 官方 scalar noise key 是 `std`，log noise key 是 `log_std`，恰好一个 `[6]`；本导出器校验后不导出采样噪声。
7. 激活白名单按 v3.0.1：`elu, selu, relu, crelu, lrelu, tanh, sigmoid, softplus, gelu, swish, mish, identity`。尤其 `crelu → CELU`、`swish → SiLU`。不进行 `eval` 或动态类导入；大写/未知名称直接拒绝。

## 3. Run manifest 协议

`--run-manifest` 是普通 UTF-8 JSON object，必填字段如下；下面 hash 仅为格式占位示例，**不得当作真实训练 hash**：

```json
{
  "schema_version": 1,
  "contract_id": "own-v40-jointspace-h5-v1",
  "contract_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "asset_manifest_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "actor_obs_dim": 125,
  "critic_obs_dim": 29,
  "action_dim": 6,
  "policy": {
    "class_name": "ActorCritic",
    "actor_hidden_dims": [256, 128, 64],
    "critic_hidden_dims": [256, 128, 64],
    "activation": "elu",
    "empirical_normalization": false
  }
}
```

- Hash 必须是 **64 位小写十六进制**，其实际计算规则由父合约/资产清单生成器统一负责；checkpoint 与 manifest 必须用同一规则和同一实际版本。
- 顶层允许额外 primitive JSON provenance 字段，不能携带启用的 normalization 或未支持的 normalization state。
- `policy` 只允许上面五项，外加可选的 `actor_obs_normalization: false` / `critic_obs_normalization: false`。两者不能为 true，`empirical_normalization` 不能省略或用 `0` 冒充 false。
- Hidden dims 必须是非空的正整数数组；不接受 `-1`（官方可推导，但此处禁止推导）、bool、浮点数、空数组；不从权重反推层数或架构。
- JSON 重复 key、NaN/Infinity、错误版本、错误 dimensions 均拒绝。

Checkpoint `infos` 必须是纯 primitive JSON dict，最少：

```python
infos = {
    "contract_id": run_manifest["contract_id"],
    "contract_sha256": run_manifest["contract_sha256"],
    "asset_manifest_sha256": run_manifest["asset_manifest_sha256"],
}
# runner.save(path, infos=infos)，包括自动保存路径；不能只给最后一次手动保存。
```

`infos` 不能有 Tensor、tuple、配置类、任意 module 或非有限数字。若额外记录 `schema_version / actor_obs_dim / critic_obs_dim / action_dim / policy`，也必须与 manifest 相同。

安全边界：`torch.load(..., weights_only=True, map_location="cpu")`；无 unsafe 重试，无任意模块反序列化/执行。只接受完整的 stock tensor state，精确检查全部 actor/critic 参数名称、shape、dense CPU float32、有限性，并检查 `std/log_std` shape/有限性（`std` 必须正）。不静默转 dtype、不忽略额外 actor/encoder/normalizer 参数、不只检查首末层。顶层旧 `ac` 即使同时携带正确 manifest 也拒绝。

**这些 hash 是一致性与来源关联，不是签名认证。** 三参数接口没有原始合约/资产文件，无法重新审计其内容，也无法从同 shape 的张量证明训练真的用了某个无参数 activation 或某版库。必须信任父训练写出的 manifest；部署应另外对照实际合约/资产。不要给无来源的旧 checkpoint 人工补 `infos` 以绕过门禁。

## 4. CLI 与 Python API

在训练仓根目录，用现有轻量验证环境，无需安装大包：

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=../.rl_deps_rsl23 \
../.venv_mj314/bin/python scripts/export_v40_onnx.py \
  --checkpoint /absolute/run/model_1000.pt \
  --run-manifest /absolute/run/run_manifest.json \
  --output /absolute/export/actor_v40.onnx
```

这里 `.rl_deps_rsl23` 仅提供本地 Torch/ONNX 的 import 路径，本脚本**不导入其中的 rsl_rl**。Python API：`wheeled_algo.v40_export.export_checkpoint(checkpoint, run_manifest, output)` 返回 sidecar dict；`load_actor_checkpoint` 返回严格核验的 CPU mean actor 与 provenance。

成功创建：

- `actor_v40.onnx`
- `actor_v40.onnx.json`（直接在完整输出文件名后加 `.json`）

任何一个目标已存在（包括悬空 symlink）均拒绝；没有 force 开关，请使用新文件名。先在输出目录同一文件系统内暂存，完成验证后用原子 no-replace hard link 发布。第二文件发布冲突时回滚属于本次的模型，不覆盖并发写入者。文件系统必须支持 hard link；不支持时失败，不能降级成可能覆盖的 rename。

这不是跨两个文件的原子事务；进程被强杀可能留下单文件，部署必须要求两文件齐全并验证 `onnx_sha256`。异常导出不会发布未通过数值校验的模型。

## 5. 验证与 sidecar

ONNX checker full check、固定 I/O 名称/dtype/shape、标准 opset17、自包含权重检查之后，必须在 **ONNXRuntime CPUExecutionProvider** 执行：

- 一个全零输入；
- `seed=40` 的 NumPy PCG64 标准正态输入 8 个，scale `[0.1,1,3,10]` 重复两次；
- 各输出检查 `float32[1,6]` 和有限性，再逐样本比较 CPU Torch：`atol=1e-6, rtol=1e-5`。

缺 ONNXRuntime、输出 NaN/Inf、shape 错误或数值不一致都失败，不生成“待验证成功”包。测试输入只是数值检验，**不是物理观测有效域或任何机器人仿真**。

Sidecar 包含 checkpoint 原始文件 SHA、run manifest 原始文件 SHA、contract/asset SHA、ONNX SHA，完整 manifest、网络配置、输入输出 shape、两种动作名、opset、版本、seed、逐样本与总体最大绝对/相对误差、容差和预处理边界。相对误差分母下限 `1e-6`，不能单看 relative max 判断失败；通过标准是逐元素 `atol + rtol*abs(torch)`。

预处理在环境/部署外部完成：

- 每帧25D顺序：角速度3、重力方向3、`[vx,wz,height]`3、四腿相对角4、六关节速度6、上一策略输出6。
- 5帧 oldest→newest，当前帧在最后25项；固定单位/坐标转换/缩放/裁剪、reset 填帧、上一动作更新必须遵循被 hash 绑定的合约。
- ONNX 只执行 MLP，**不内嵌 history buffer、经验归一化、控制器、电机限幅或安全监控**。
- ONNX/GEMM 本身不会拒绝 NaN 输入；调用方必须在推理前后验证 shape/dtype/有限性。模块提供 `validate_observation` 作 CPU Torch 侧同等检查。不要声称一个无校验的裸 ORT session 可以自动拒绝 NaN。

## 6. 纯导出回归

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src:../.rl_deps_rsl23 \
../.venv_mj314/bin/python -m pytest -q -p no:cacheprovider tests/v40/test_export.py
```

测试仅构造随机初始化的合法小 checkpoint 和项目默认 MLP，检查 key/metadata/hash/shape/安全加载、激活白名单、拒绝归一化、实际 ONNX/ORT parity、独立随机输入、错误版本/35D/V3/ac、NaN、不可用 ORT、失败回滚、已有文件/发布竞态与 CLI。所有运行时 checkpoint/ONNX 在 pytest 临时目录；不修改旧脚本/pyproject，不进行训练，不将通过测试写成策略有效或目标 Isaac/RSL 3.0.1 集成已验收。

本轮实际执行：**122 passed，3.72s**。使用现有 Torch `2.14.0+cu130`、ONNX `1.22.0`、ONNXRuntime `1.27.0`、pytest `9.0.3`。全部 12 个白名单激活都执行了真实 ORT 校验；合成 `[256,128,64]` 默认网络的9个导出校验输入最大绝对误差为 `3.5762786865234375e-7`。50条 warning 来自要求的 `dynamo=False` TorchScript 导出器弃用提示，不是测试失败。上述结果仅属于纯序列化/数值回归。
