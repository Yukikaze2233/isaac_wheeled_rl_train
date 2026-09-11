# V40 Isaac Lab 执行入口（拒绝式预检）

## 状态与边界

目标是 **Isaac Lab仓库tag v2.3.0 / Isaac Sim5.1.0 / RSL3.0.1**。该Lab tag内部发行包为isaaclab0.47.2、assets0.2.3、tasks0.11.6、rl0.4.4；入口核对这四个版本和官方真实commit `3c6e67bb5c7ada942a6d1884ab69338f57596f77`，不把仓库release误作pip包版本。本文是代码接口说明，本地测试不含Isaac；当前服务器进度须看各阶段运行报告。已完成最小无相机框架验证不等于V4模型已通过或已训练。

`--help` 与 `--preflight-only` 在 `AppLauncher` 之前运行，不导入 Isaac。所有入口都核对实际分发版本、core 合约与资产；缺包、版本不匹配、未批准碰撞、硬件先验未获授权等均列入 `blockers`，退出码 **2**。预检不建日志目录、不转换 USD、不运行物理。

目标解释器还必须具备隔离 CPU 导出依赖 **`onnx==1.20.1`、`onnxruntime==1.20.1`**，
后者要求 CPU distribution `onnxruntime`，不能用 `onnxruntime-gpu` 的 metadata 代替。
共享预检仅通过 `importlib.metadata.version` 核对版本，缺失/不符在 Sim 启动前拒绝；
不在预检或活跃 Kit 内导入 ONNX/ORT。metadata 通过不证明原生库可用，实际导出与 CPU ORT
数值对齐仍由既有隔离子进程验证，入口不会自动安装依赖。

当前默认契约已选择用户明确批准的 `research_manifest.json`：限定研究碰撞范围的静态门为true；原 `manifest.json` 的源材料审查仍false且字节未改。`--research` 仍必须显式提供，不能代替审批记录、source哈希、六对过滤范围及软件/运行检查。研究批准不代表CAD已修复或hardware_deployment_ready，不能手改一个passed标志伪造准入。

四个入口共用 `train_v40.launch_app`，仅在全部预检通过且实际启动时设置
`ENABLE_CAMERAS=0`、`LIVESTREAM=0`，并显式传 `enable_cameras=False`、`livestream=0`。
这是针对已只读核对的 Lab v2.3.0 环境继承语义；root 进程另传 `kit_args="--allow-root"`，
不修改 Lab。help/纯预检不改调用者环境、不导入 AppLauncher；训练预算仍在其 import 前后检查。
四入口 cleanup 均保证 env.close 出错后仍尝试 app.close。

## 代码与冻结接口

- 环境：`src/wheeled_tasks/direct/v40_serial/{__init__.py,env_cfg.py,env.py}`；注册名 `Own-V40-Serial-Direct-v0`。CLI 直接创建 V40 类，不选择旧任务。
- 资产：`src/wheeled_world/assets/v40.py`，只加载通过 core 验证的 canonical URDF，X 前/Y 左/Z 上；**不再次旋转**根姿态或观测。
- PPO：`src/wheeled_tasks/agents/v40_ppo_cfg.py`，stock `OnPolicyRunner` / `PPO` / `ActorCritic`，不含自研 PPO、GRU、encoder、aux。
- 观测：policy 为 oldest→newest 的 **5×25=125D**；critic 为当前25D + body线速度3 + 离本环境地面高度1 = **29D**；动作6D。
- v2.3.0 `DirectRLEnv` 会将 `observation_space` 包进 `policy`，将 `state_space` 包进 `critic`，故配置分别为 **125 / 29**，不是再嵌套 dict。环境返回顶层 `{"policy": N125, "critic": N29}`；官方 wrapper 构造 TensorDict。runner 显式使用 `obs_groups={"policy":["policy"],"critic":["critic"]}`。
- actor/critic hidden dims 均 `[256,128,64]`，ELU；两侧 observation normalization 都显式 false，deprecated empirical normalization 也为 false。
- physics dt=.005、decimation=2、policy dt=.01；rollout48（0.48秒），SCUT普通PPO起点：lr1e-4、adaptive KL.01、gamma.99、lambda.95、5 epochs/4 minibatches、clip.2、entropy.005、valuecoef4。上限20,000为本次追加迭代上限，不是时长/收敛保证；首测显式2/100次。noise std.2配本机wheel scale50得到初轮速std10rad/s，未直接抄SCUT的std1。没有自动高速课程或重域随机化。

## 执行时序与终止

1. `_pre_physics_step` 保存上一动作作 action-rate 项，调用 core `decode_targets` 得四腿位置目标、两轮速度目标及已裁剪动作。
2. 每个 `_apply_action` **重新读取当前关节反馈**，调用 core `compute_torques` 并输出 efforts。导入器 `JointDriveCfg(target_type="none")` 和 `IdealPDActuatorCfg(stiffness=0,damping=0)` 均关闭 PD，不能双重施加 PD。
3. `IdealPD` 的 effort/armature 显式来自当前合约，wheel torque-speed 包线由 core 在正确轴域处理。URDF 的占位 `velocity=1` 不作为实际电机额定值：solver velocity bound 设为 1e9 以取消这项人为刹车，**1e9 不是电机速度规格**。目前缺少实测腿部速度包线，仍属于 research 限制。
4. step 后先用旧命令计算奖励与实际跟踪误差，再标记命令到期；只在下一 `_get_observations` 前采样新命令，防止动作依据旧目标、奖励依据新目标。
5. 返回观测中的 last action 是刚执行的 clipped `a_t`，不是 `a_(t-1)`。HistoryStack 按 `common_step_counter` 更新，同 tick 重复查询不重复推进；reset 后首帧重复填满5帧。
6. reset 只清指定 env 的动作、前动作、力矩、targets、历史、命令、命令时钟及 episode 累计；根状态复制 default 后加 `env_origins`，保留 default quaternion。
7. terminated 分别检查非有限状态/动作、非轮触地、膝硬界、倾角>35°、高度<.18m，以及基座 visual 包围盒八角点保守净空≤0；timeout 与 terminated 分开，`is_finite_horizon=false`，由官方 wrapper 把 timeout 放入 `extras["time_outs"]` 供 stock PPO bootstrap。
8. 已读取实际 `core.compute_reward_terms`：返回的非 terminal 各项已乘权重和 policy_dt；环境直接相加，**不再乘 dt 或重复加权**。terminal event penalty 不乘 dt。每项独立写 `Reward/*`、episode每秒项，接触 `Contact/*`，实际速度/高度/误差 `Tracking/*`，原因 `Termination/*`。零命令平动惩罚来自 core，不将“无 vx 指令”当作允许持续漂移；没有全局 XY 位置保持保证。

阶段由 `--stage` 显式选择，不自动晋级：stand `[0,0,.32]`；height 高度 `.28..32`。
v1 locomotion 为vx `±.5`、wz `±1`；当前v2主任务为vx/wz `±2`、高度 `.28..32`，
每次重采样以10%概率将两个速度命令同时置零，用同一个策略学习站立和移动。
命令切换保留历史，固定命令评估绕过随机采样。命令范围与周期读取实际 JSON，不用旧任务默认值。

## 接触、关节与 USD：尚需真实 Isaac 验证

环境注册 `scene.articulations["robot"]`、`scene.sensors["contact"]`，创建地面并 clone，CPU/GPU 均显式过滤跨环境碰撞。为逐环境编辑/读回 USD 窄过滤，当前使用 `replicate_physics=False`、`clone_in_fabric=False`、`copy_from_source=True`；这是可审计性优先的保守配置，不是性能最优声明，恢复复制优化前需单独验证。保留 self-collision，不做全局关闭。URDF 的7个 body 为 base、左右各3段；sensor 同时覆盖全部7个 body，初始化核对数量与名称。

动作顺序固定 `[L_joint1,L_joint2,L_joint3,R_joint1,R_jonit2,R_joint3]`。实际导入索引按名字逐个查找，转为 `torch.long`；不依赖正则匹配排序。左右轮按 `L_link3/R_link3` 取力；非轮为其余5个 body，**绝不以 base_force 代表轮接地**。

使用 `ContactSensor.data.net_forces_w_history`（预期 N×history×7×3）覆盖两个 physics substep，`net_forces_w` 预期 N×7×3。这里是 rigid-body **net normal contact forces**（不含切向摩擦力），可能含自碰撞，向量抵消也可能影响阈值，不是地面对轮的专属接触力；日志因此叫 `wheel_contact_candidate`。ground pair 过滤、importer body path、reset 后 force 缓存、CPU/GPU history 更新时序、真实力方向等必须在目标 Sim 验证，不能将 AST 通过写成“接地 API 已验收”。

URDF 直接经目标 tag 的 `UrdfFileCfg/UrdfConverterCfg` 转换；`target_type="none"` 是 v2.3.0 支持的字段。逐块 collision mesh 使用 convex hull，不由 visual 重新生成或静默改用整机 hull。转换输出在 `logs/v40_usd_cache/<asset_manifest_sha256>/`；强制重新转换避免旧 USD 缓存。PhysX 实际烹饪精度、相邻 link 过滤、关节限位方向、armature/effort应用及高扭矩步进响应仍需服务器验证；离线 collision passed 也不是动态工作域验收。原清单双髋材料仍pending。研究变体另带 `research_model_approval.json` 与原清单SHA，保留原geometry_review_supported值，研究status为 `user_approved_joint_internal_contact_exclusion`。URDF不会自动携带SRDF。`apply_approved_collision_filters`在clone后按实际RigidBody解析，统一调用core.validate_filter_policy；研究分支必须传入由audit_asset核验的approval/raw记录，验证明确用户决定、所有原文件不变、精确6对及真实静态检查；原raw分支仍要求geometry支持。未知/重复对、缺失/重名刚体、未授权或跨env过滤均在USD编辑前拒绝，不能仅改passed或抹原材料失败。随后逐 env 应用 `UsdPhysics.FilteredPairsAPI`，每对写双向 targets（6对/12 targets），精确读回数量与路径；不靠 importer 的隐式行为替代显式关系。`check_v40_env` 的有限检查结果包含 `collision_filter_usd` 计数和作用域。USD 写入/读回不是 PhysX 实效验收，不写 server_verified 标记；默认相邻关节碰撞语义、独立克隆过滤效果仍需目标服务器验证。

基座 `base_visual_bounds_m` 是 canonical **base-link 局部 visual AABB**，不是 COM。环境将其8角点按实际根四元数旋转，取最低 world-z 并减本环境地面高度，记录 `Geometry/base_visual_clearance_lower_bound_m`。这是包围盒对真实 mesh 净空的保守下界：≤0 可保守提前终止，可能误报，**不是精确 mesh 接触/距离或质量中心位置**；原 .18m base-origin 高度门仍独立保留。

### Continuous 导入硬停根因修复

服务器修复前实测：四个 hips/wheels 的 `root_physx_view.get_dof_limits()` 为
`[-3.1399999,+3.1399999] rad`，对应 USD 约 `±179.908°`；两膝正确。
旧 final policy 在约 7.04 s 失败、轮角接近 `±3.135 rad`。这暴露的是物理导入错误：
source/canonical URDF 虽标记 `type="continuous"`，仍保留占位
`<limit lower="-3.14" upper="3.14" effort="100" velocity="1">`，Isaac 导入器将其转换成硬停。
离线 MJCF generator 会按 continuous 类型设置 `limited="false"`，原 URDF 门禁也只核验类型，
所以静态门通过不能捕获实际 solver 的错误限位。

修复位于 `src/wheeled_world/assets/v40.py` 的 `normalize_v40_usd_joint_limits`：

- 使用 `DirectRLEnv` 拥有的 `sim.get_initial_stage()`，在 `_setup_scene` clone 后、
  父类首次 `sim.reset()` 激活物理之前，逐 clone 精确解析六个关节。
- 四个 continuous 关节仅通过 **`UsdPhysics.RevoluteJoint.CreateLowerLimitAttr/ CreateUpperLimitAttr`**
  写入 `(-inf,+inf)` 并读回 composed USD。USD 的单位是度；这是该 typed schema 的无界语义，
  不使用大有限角度，也不新增/猜测 `limitEnabled` 属性。
- 两膝只读核对：`L_joint2=[-0.12672741539177768,0.6586707480056704] rad`、
  `R_jonit2=[-0.6119707480056704,0.17342741539177764] rad`，保留真实内角 **35–80°**。
  USD float32 度→弧度允许 `1e-6 rad` 数值误差，`rtol=0`；不写膝限位。
- 缺失/重名/额外关节、非 revolute schema、额外 `PhysicsLimitAPI:*`、膝值不符在编辑前拒绝；
  不删除 `PhysxLimitAPI` 等物理参数 schema。写入失败或 composed readback 不符直接抛错。
- `V40Env` 在父类初始化完成后、首次用户 reset/rollout 前调用
  `check_physics_joint_limits()`：共享 `validate_v40_physx_joint_limits` 按实际 joint name 查询
  **真实 `root_physx_view.get_dof_limits()`**，校验完整 `[num_envs,6,2]`，所有 clone 均须
  四关节为成对 `(-inf,+inf)` **或精确的** `(-FLT_MAX,+FLT_MAX)`，两膝有限且匹配。
  服务器诊断 `/tmp/opencode/v40-limit-probe.log` 已确认 composed USD 为 `(-inf,+inf)` 时，
  PhysX 返回 `±3.4028234663852886e38`，即 `torch.finfo(torch.float32).max` 的原生 PxReal
  无界范围编码；先前仅接受 IEEE infinity 的检查器假设错误，USD 物理修复本身已正确。
  sentinel 使用精确相等，不用阈值或近似比较；±π、±1e10、邻近 FLT_MAX 的有限值、
  单边界、混合 infinity/FLT_MAX 编码或 NaN 都失败，FLT_MAX 也不能替代膝限位。
  不以 `data.soft_joint_pos_limits` 或配置值代替物理证据；Lab 2.3.0 的 soft-limit 均值公式
  对 `(-inf,+inf)` 可产生 NaN，该缓存不参与本 V40 task 的控制/观测/限位判定。

`check_v40_env.py --apply --research` 额外在 reset 后与 bounded steps 后重新读 solver，输出
`joint_limits_usd`、`joint_limits_physx_at_reset`、`joint_limits_physx_after_steps`。
`limits_rad_env0` 保留实际有限数值（包括 `±3.4028234663852886e38` 与两膝 readback），
仅真正的 IEEE infinities 用字符串 `"-inf"/"+inf"` 表达以兼容严格 JSON；
`limit_encoding_env0` 逐关节标注 `ieee_infinity`、`physx_float32_extrema` 或 `finite_knee`。
数值/编码描述 env 0，校验覆盖所有 clone；允许不同 clone 使用上述两种合法成对编码。

这是运行 stage 的导入边界修复。canonical/source URDF、effort/velocity 原文、原始资产、
raw/research manifests、审批及历史证据的字节与 hash 保持原样；无须重新生成或补造审批。
强制 USD conversion 仍按原配置运行，每次导入都重新应用 stage override；缓存文件本身不是修复后的
composed stage。资产 identity hash 相同不代表修复前后仿真行为相同，复测必须绑定**新代码快照**。

已阅读仓内四份官方参考 `docs/reference/isaaclab_v2_3_0/{cartpole_env.py,vecenv_wrapper.py,rl_cfg.py,official_train.py}`，并额外只读核对官方 tag 的 [DirectRLEnv](https://raw.githubusercontent.com/isaac-sim/IsaacLab/v2.3.0/source/isaaclab/isaaclab/envs/direct_rl_env.py)、[URDF converter cfg](https://raw.githubusercontent.com/isaac-sim/IsaacLab/v2.3.0/source/isaaclab/isaaclab/sim/converters/urdf_converter_cfg.py)、[UrdfFileCfg](https://raw.githubusercontent.com/isaac-sim/IsaacLab/v2.3.0/source/isaaclab/isaaclab/sim/spawners/from_files/from_files_cfg.py)。这只是源码 API 核对，不是已导入目标库。

## 命令示例（本次未执行任何仿真/训练）

仓根，任意 CPU Python 可以查看 help：

```bash
python scripts/train_v40.py --help
python scripts/play_v40.py --help
python scripts/check_v40_env.py --help
python scripts/check_v40_env.py  # 默认严格预检；当前阻断时 exit 2
python scripts/train_v40.py --preflight-only --research --num_envs 1
```

纯预检从父实现的 stdlib-only `wheeled_tasks.v40.contract` 读取合约/资产 API，不依赖 Torch 或 Isaac；目标 runtime 包缺失同样明确阻断，而非“跳过后通过”。checkpoint 数值校验在前述门禁通过之后使用已有 CPU Torch，任何缺失都不会尝试自动安装。

以下仅在审批后的目标 Isaac 环境使用，`<ISAACLAB>` 是已有的官方 checkout：

```bash
PY=/absolute/isaac-env/bin/python
"$PY" scripts/check_v40_env.py \
  --apply --research --headless --num_envs 1 --max_steps 20
"$PY" scripts/play_v40.py \
  --research --headless --checkpoint /absolute/new-v40-run/model_final.pt --num_envs 1 --max_steps 1000
```

实际训练按用户要求必须由独立tmux启动：从本机使用 `scripts/start_v40_tmux.py`，明确绝对环境解释器、远端仓/新run目录/独立audit目录、时区化stop-at及短预算；见 `V40_TIMED_RUN.md`。不要依赖tmux服务器旧PATH，也不要在SSH前台直接开长训。

`check_v40_env` 仅同时显式 `--apply --research` 才可能启 Sim；num_envs 最大2、max_steps最大40，超界直接拒绝，不暗中截断。只施加零归一化动作诊断接线，不调用 runner、不训练；任何 terminated 即停止，异常/未完成 bounded check 为非零退出。

### 长训前：新服务器代码快照的限位 / shape / 多圈检查

先在**新快照仓根**执行，保存 stdout/stderr 到新 audit 目录并记录代码快照身份，不回写冻结旧 run：

```bash
"$PY" scripts/check_v40_env.py --apply --research --headless --num_envs 2 --max_steps 40
```

要求退出 0、policy `[2,125]` / critic `[2,29]` 有限、solver limits `[2,6,2]`，
两个 readback 均显示四 continuous 无界、两膝正确。此检查只有 0.4 s、零动作，**不是多圈验收**。

下面是独立、有限的物理多圈诊断（总计 80 physics steps = 0.4 s），复用同一资产门禁、
adapter 与 solver checker。将两个自由基座一次性放在空中，给两轮一次性初速 ±50 rad/s，
随后零 effort 自由演化；重力、碰撞与膝硬界保留，不锁基座、不循环重写轮角或速度。
这里的初速是诊断激励，不是新增电机先验。直接 step physics 以避免策略/任务 auto-reset
掩盖轮角；不调用 policy/reward/termination，不证明地面平衡或训练质量。

```bash
PYTHONPATH="scripts:src${PYTHONPATH:+:$PYTHONPATH}" "$PY" - <<'PY'
import argparse
import json
import math
from train_v40 import add_common_arguments, preflight, print_preflight, launch_app, make_env

parser = argparse.ArgumentParser()
add_common_arguments(parser)
args = parser.parse_args(["--research", "--headless", "--num_envs", "2"])
report, _, _ = preflight(args)
print_preflight(report)
if not report["ready"]:
    raise SystemExit(2)
launcher = launch_app(args)
env = None
try:
    import torch
    env = make_env(args)
    env.reset()
    robot = env.robot
    wheel_ids = [robot.joint_names.index(name) for name in ("L_joint3", "R_joint3")]
    root = robot.data.default_root_state.clone()
    root[:, :3] += env.scene.env_origins
    root[:, 2] += 2.0
    root[:, 7:] = 0.0
    q0 = robot.data.default_joint_pos.clone()
    dq0 = torch.zeros_like(q0)
    directions = torch.tensor([1., -1.], device=env.device)[:, None]
    dq0[:, wheel_ids] = directions * 50.0
    robot.write_root_pose_to_sim(root[:, :7])
    robot.write_root_velocity_to_sim(root[:, 7:])
    robot.write_joint_state_to_sim(q0, dq0)
    robot.set_joint_effort_target(torch.zeros_like(q0))
    samples = []
    previous_q = q0[:, wheel_ids].clone()
    travel = torch.zeros_like(previous_q)
    velocity_integral = torch.zeros_like(previous_q)
    for step in range(80):
        if not launcher.app.is_running():
            raise RuntimeError("simulation stopped before completing multi-turn check")
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)
        q = robot.data.joint_pos[:, wheel_ids].clone()
        dq = robot.data.joint_vel[:, wheel_ids].clone()
        assert torch.isfinite(robot.data.joint_pos).all()
        assert torch.isfinite(robot.data.joint_vel).all()
        # PhysX wraps reported joint angles; resolve only sub-Nyquist increments.
        assert bool((dq.abs() * env.physics_dt < math.pi / 2).all())
        travel += torch.atan2(torch.sin(q - previous_q), torch.cos(q - previous_q))
        velocity_integral += dq * env.physics_dt
        previous_q = q
        samples.append({"time_s": (step + 1) * env.physics_dt,
                        "wheel_q_rad": q.tolist(), "wheel_dq_rad_s": dq.tolist()})
    torch.testing.assert_close(travel, velocity_integral, atol=1e-4, rtol=1e-4)
    travel = travel * directions
    limits = env.check_physics_joint_limits()
    print(json.dumps({"solver_limits": limits, "signed_travel_rad": travel.tolist(),
                      "samples": samples}, allow_nan=False))
    assert bool((travel > 4 * math.pi).all()), "each wheel must traverse >2 full turns in its test direction"
    assert bool((dq * directions > 0).all()), "wheel stalled or reversed"
finally:
    try:
        if env is not None:
            env.close()
    finally:
        launcher.app.close()
PY
```

验收需同时有 solver readback 与连续运动轨迹：每个方向、每只轮都越过 ±π 并累计超过两圈。
PhysX 会回绕报告的无界关节角，故终值减初值不是圈数；在每步角变化远小于 π 的激励下展开
连续角增量，并与实测关节速度积分交叉核验。不能用 teleport 或多次 reset 拼接代替。
失败时保留日志定位，不调整奖励来掩盖。
通过后再用现有 play/evaluate 入口对旧 checkpoint 做新快照下的独立评估（尤其原 7.04 s 之后）；
多圈通过本身不承诺旧 policy 会成功，也不代替长训前的接触/shape 检查。

## Checkpoint 身份、resume 与 finetune

run 目录必须全新（包括 symlink/已有目录均拒绝），不接续写回旧目录。`run_manifest.json` 使用 core `make_run_manifest`，通过现有严格 exporter 的 schema 验证，额外记录 stage/seed/research/目标版本；同时保存 agent_config。checkpoint 与运行时当前 contract/asset 三项 identity 必须完全匹配。

所有 `runner.save` 都在调用 `learn` **之前**绑定 metadata adapter：只包装 stock 保存方法，不替换 PPO 或 runner 训练实现。因此自动中间保存、stock learn 尾部保存、显式 `model_final.pt` 保存均含 `infos={contract_id,contract_sha256,asset_manifest_sha256}`。外来 infos 不能覆盖 identity，非 JSON/NaN 被拒绝。

```bash
# 恢复模型、optimizer、iteration；写入另一全新目录
... scripts/train_v40.py --research --resume /old/run/model_100.pt --run-dir /new/resume-run
# 仅恢复 stock model_state_dict；新 optimizer，iteration=0
... scripts/train_v40.py --research --finetune /old/run/model_100.pt --run-dir /new/finetune-run
```

两参数互斥；都先用现有 `load_actor_checkpoint` 严格核验完整 actor/critic 权重与 side-by-side run_manifest。恢复使用 `torch.load(weights_only=True)`，无 unsafe 重试；不调用可能采用宽松 unpickle 的路径。resume 不声称恢复环境状态/RNG或 bit-exact 接续；`max_iterations` 是本次新增学习迭代数。finetune 当前必须使用同一物理合约及资产 hash，可显式换训练 stage，不容许跨机器人权重“碰巧维度一致”就加载。

play 复用 exporter 的严格 CPU mean actor，核验当前 identity 后转指定 device 做 bounded deterministic rollout，不随机动作，不重新导出。既有 `scripts/export_v40_onnx.py` 与 `docs/V40_EXPORT.md` 未修改。

训练自动导出使用已有隔离 CPU 子进程，不在活跃 Kit 内调用 ONNX/ORT：保留启动前环境/cwd，
同一绝对解释器运行 `export_v40_onnx.py`，隐藏 CUDA；父进程检查退出码、sidecar 和哈希后
才发布完成收据。导出日志、超时和失败时保留 pt 的细节见 `V40_TIMED_RUN.md`。

## 本地验证范围

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=src:../.rl_deps_rsl23 \
../.venv_mj314/bin/python -B -m pytest -q -p no:cacheprovider \
  tests/v40/test_launch.py tests/v40/test_job.py tests/v40/test_env_contract_static.py \
  tests/v40/test_metrics.py tests/v40/test_export.py tests/v40/test_joint_import_limits.py
```

历史 **37 passed（1.53s）** 属于早期接线测试，不代表当前研究模型或服务器集成结果。
当前默认研究清单与原材料失败清单须区分；目标 runtime 缺失/不匹配仍独立阻断。
启动回归使用 stub AppLauncher/环境变量验证 root 参数、无相机/直播、惰性导入及预算边界；
tmux 测试仅执行全部 subprocess mocked 的序列化脚本，覆盖多 argv 和旧服务端环境。

测试覆盖新文件 AST、CPU help、受限 check 参数、PPO字段、保存 metadata 钩子、research 不绕碰撞、源码中的 scene/history/reset/PD 接线。另从 AST 单独提取奖励/观测方法，接入真实 CPU core：验证 nominal stand reward 为 .05（不是重复 dt 后的 .0005）、空/部分有限批次、terminal penalty 不缩放、命令晚采样、29D critic、last action、历史快照不被下一步修改、同 tick subset reset。窄过滤用纯 fake-stage/schema 验证全计划先审查、pending 不写入、双 env 命名与12targets读回、未知/跨环境过滤拒绝、幂等与失败读回；八角点公式用真实 CPU Torch 测试旋转及 env ground 偏移。**没有在本机导入 pxr/Isaac，也不把这些接口单测冒充集成**，不证明 PhysX 能运行或策略稳定。core 的张量算法与历史测试由独立 `tests/v40` 用例负责；目标服务器上的接触/转换/向量化 step 和 checkpoint 集成验证仍是交付后的必要关卡。

导入限位新增回归：`tests/v40/test_joint_import_limits.py`。fake USD 覆盖双 clone、float32 度转换、
四 continuous 写入及读回、膝/无关 prim/其他参数不变、幂等、全部 clone 先验证、schema/写入/读回失败；
真实 CPU Tensor 覆盖乱序 joint name、所有 clone 的 ±3.14 复现、大有限界/单边界/NaN、膝值/shape 不符，
并核验真实磁盘资产 hash、原审批状态与 source/canonical URDF 的 effort/velocity 字段保留。
这些测试不导入 Isaac/pxr，不代表上述远端多圈诊断已经执行。

本次完整 V40 CPU 回归命令及结果：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=src:../.rl_deps_rsl23 \
/home/yukikaze/Documents/workspace/robot_rl/.venv_mj314/bin/python -B \
  -m pytest -q -p no:cacheprovider tests/v40
# 522 passed, 50 warnings in 46.02s (existing ONNX export deprecation warnings)
```

本地 `check_v40_env.py --preflight-only --research --num_envs 2` 正常拒绝缺失/不匹配的目标 runtime，
退出 2，`simulation_started=false`；完成资产 hash 审计，未转换 USD 或启动物理。
