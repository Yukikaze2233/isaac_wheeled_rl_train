# V6 1kHz控制与独立设计预检接线

状态：工程实现及离线测试；**没有发布新资产训练合同，没有启动PPO，也没有通过G1或通信功能验收。** 本文接续`V6_NEW_ASSET_TRAINING_MASTER_PLAN_20260928.md`，不替代其预算、资产、能力和冻结要求。

## 已实现的边界

`src/wheeled_tasks/chassis/preflight.py`不依赖Isaac/torch。`control_decimation`拒绝非正、非有限或非整数比例时钟，不再将错误比例静默四舍五入。旧`.005/.02`仍合法；新设计表schema限定`.001/.02`、腿160/2.5及轮速度P0.6。这里轮prior旧字段`kd`实际表示速度P。

`env.py`仍使用现有显式执行器路径：策略周期开始解码一次目标，随后每物理子步读取新`joint_pos/joint_vel`、重算PC力矩、写入执行器、执行`sim.step`、`robot.update(dt)`。已增加调用顺序及次数断言：1次目标解码、20次PD、20次物理步后反馈刷新。不是在200Hz物理步内对同一反馈调用5次PD。若显式选择旧USB反馈延迟模型，PD输入仍依该模型延迟；计数不能把延迟反馈证明为即时反馈。

启动报告新增`control_timing`；首次step前标记`pending_first_step`，完成后保存实际调用计数与周期。计数只证明执行链和仿真时间配置，不表示CPU墙钟实时达到1kHz，也不是PhysX/新资产闭环能力验收。无需增加GPU同步或复制每帧张量。

`validate_design_contract`按独立表的stage和run role比较最终配置：关键字段即使恰好等于旧合法默认，删除后也拒绝；嵌套字典缺项、多项、范围变化均报告字段路径、设计值和有效值。USB旧双向模型与新下行模型互斥。训练seed可声明`root_plus_block_index`，不能把派生block seed当成独立训练seed。评测seed与实际调用seed必须一致。

`design_preflight`从绑定的独立文件读取设计表并核对SHA。新合同必须显式设置`requires_design_preflight: true`；绑定缺失拒绝。未启用此机制的旧合同返回`None`，**不登记设计预检通过**。此兼容路径不允许用于宣称新主线已冻结；正式启动器还须按新schema强制要求该标记，防止同时删除标记与绑定绕回旧入口。

环境构造在创建SimulationContext前调用统一静态检查；初始化后核对运行时模块实例、配置时钟、实际环境generator初始seed及控制增益。训练和评测源码归档均加入了新模块。`validate_runtime_state`的回执明确记录功能探针`not_run`。

## 设计表schema示例：不可执行

以下只是接口示例，`status`故意不是`frozen`，包含未决值，不得改个状态后直接启动。真实表由G0/G1、奖励定标和能力表独立冻结，不能从待验证物化合同反生成。

```json
{
  "schema": "chassis-design-expectations-v1",
  "status": "example_nonexecutable",
  "stages": {
    "S1": {
      "roles": {
        "train": {
          "contract": {
            "curriculum_stage": "S1",
            "enabled_stages": ["foundation"],
            "stages": null,
            "evaluation": null,
            "physics_dt": 0.001,
            "policy_dt": 0.02,
            "v5_control": {"leg_kp": 160.0, "leg_kd": 2.5},
            "usb_transport": {"enabled": false},
            "command_transport": null,
            "precision_tracking": null,
            "signal_perturbations": null,
            "dynamics_randomization": null,
            "contact_domain": null,
            "step_assist": {"enabled": false},
            "command_reference": null,
            "reference_reward": null,
            "reward_profile": null,
            "actor_dim": 35,
            "critic_dim": 81,
            "action_dim": 6,
            "history_length": 1,
            "actor_layout": null,
            "critic_layout": null,
            "asset_manifest_sha256": null,
            "target_num_envs": 20000,
            "num_steps_per_env": 24,
            "num_mini_batches": 10,
            "num_learning_epochs": 5,
            "total_updates": 1000,
            "evaluation.seed": null,
            "evaluation.confirmation_seed": null
          },
          "control": {"actuators.wheel": null},
          "invocation": {
            "num_envs": 20000,
            "seed": {"rule": "root_plus_block_index", "root_seed": null}
          },
          "runtime": {
            "physics_dt": 0.001,
            "policy_dt": 0.02,
            "decimation": 20,
            "pc_control_dt": 0.001,
            "seed": {"source": "effective_seed"},
            "modules": {
              "usb_transport": false,
              "command_transport": null,
              "signal_perturbations": null,
              "dynamics_randomization": null,
              "contact_domain": null,
              "step_assist": false
            },
            "leg_kp": 160.0,
            "leg_kd": 2.5,
            "wheel_velocity_p": 0.6,
            "precision_tracking": true
          }
        }
      }
    }
  }
}
```

`contract`和`control`键采用点分字段路径；其值为字典时比较完整字典。真实`v5_control`及`actuators.wheel`必须包含全部生效字段，包括动作映射、参考端、限幅与曲线，不能仅复制上例两个增益。所有模块明确填写`enabled`，关闭也须声明。实际关闭precision的对照必须独立冻结为该角色/分支设计，不靠遗漏字段实现。

所有S1–S6及E0各运行臂应物化`train/baseline/quick/full/confirmation`角色行；不存在的行拒绝，没有从train继承评测默认值。评测用`{"rule":"fixed","value":实际种子}`，并将角色合同的`evaluation.seed`设为该实际种子。最终评测集、每case定义、辅助与随机化等可作为额外点分路径完整冻结。`runtime.seed`的`effective_seed`引用只引用已完成静态核验的调用seed，用于跨block而不改写设计表。

可执行合同绑定形状如下，仍仅为示意：

```json
{
  "requires_design_preflight": true,
  "design_preflight": {
    "file": "contracts/尚未发布的独立设计表.json",
    "sha256": "尚未冻结",
    "stage": "S1",
    "block_index": 0
  }
}
```

调用接口：

```python
receipt = design_preflight(
    effective_contract, actual_control_prior, ROOT,
    stage=environment_stage, role=run_role,
    seed=effective_seed, num_envs=actual_count, block_index=actual_block_index,
)
```

监督器应在baseline与训练进程生成前调用；CLI/恢复sidecar合并后重新调用。训练CLI已在`--preflight-only`返回前调用同一函数并输出回执，环境构造再次验证。评测构造`FixedCaseEnv`时传入`run_role`；默认固定评测角色为`full`，不能把baseline/quick/confirmation都假装成full。本轮未改supervisor语义、自动生成正式设计表或编造新正式合同，完整五角色静态入口接线待新可执行合同一起完成。RNG恢复后还须核验恢复来源与状态SHA；初始化generator seed不证明恢复后随机流身份。

## 已验证与剩余验证

离线测试覆盖关键字段逐项删除、设计侧漏项、嵌套配置漂移、USB互斥、固定/派生seed、独立文件哈希、错误时钟、错误运行时模块和子步顺序。测试直接执行生产`env.py`的物理循环，用每步变化的假物理反馈检验PD调用看到的新q及变化力矩；没有导入Isaac或启动GPU。

后续正式放行仍必须完成：

- 新资产Isaac smoke记录实际物理周期、目标保持、执行器力矩及20,000环境资源；与更细步长回放比较收敛。本轮代码计数不替代这项实测。
- USB/downlink确定性探针：零延迟、已知亚毫秒延迟、丢包及burst输入下逐tick核对实际输出；启用/关闭掩码和case覆盖分别验证。旧RTT模型与下行模型不能互换；当前`can_period_ms=1`下`.001`物理步应为一个通信tick，不再采用5tick区间力矩平均。USB关闭smoke不计USB功能验收。
- precision以固定反例轨迹重算实际reward项、系数和相位掩码；当前实例/配置断言不能证明奖励功能正确。源码中的硬编码系数还须随源码SHA和完整reward登记表冻结。
- 随机化以固定seed探测启用掩码、实际范围/比例、相关性及评测override；当前静态字典精确匹配不能代替抽样效果核验。现有motor/spring strength仍由signal开关关联采样，隐式范围须在新合同/发布配置中显式化。
- 资产、ABI、分阶段预算及五角色评测manifest闭包仍分别核验；当前模块不是G0/G1、功能验收、晋级或恢复来源验证的替代品。

本轮没有改actor35/critic81/MLP或六轴动作结构；没有覆盖历史合同/旧200Hz实验身份。

V6资产另接入`v6_gas_spring_closedchain_research`支持集合，训练入口使用独立`v6-gas-spring-full-stage-research-v1`合同ID；该ID强制要求设计预检标记与绑定。评测复用同一入口。旧`full_curriculum`的V5合同ID和跨资产身份比较保留。`preview_v5_springs.py`依赖旧轴向与V5生成工具，仍明确拒绝V6；不能只放开身份判断就宣称预览兼容。

2026-09-29离线验证在Kaiser的`entry_validation`独立副本执行，无GPU训练。辨识测试额外覆盖非法积分步长、记录的控制/目标频率、分段时钟、V6反号的虚功/功率不变性及系统帧力矩前向保持；这些代数测试不证明新旧质量/惯量/气簧几何等价。

## 独立20,000环境场景容量探针

已准备以下工程专用文件，未执行GPU探针：

- `contracts/v6_asset_capacity_probe_v1.json`：新V6资产、1ms物理/PC PD、50Hz目标、20,000环境，固定平地站立场景及100个零动作策略步。
- `contracts/v6_asset_capacity_control_v1.json`：独立`chassis-explicit-motor-control-v1`，腿160/2.5、±40Nm，轮速度P0.6、轮输出端±4.5Nm、比率15.8。不冒充旧V4合同或新辨识plant。
- `contracts/v6_asset_capacity_probe_expectations_v1.json`：独立工程设计表，冻结关键值、控制文件SHA和运行角色。只允许工程`train`入口角色，未声明任何评测或晋级角色。

目标为**纯物理场景容量**：通信、信号/动力学/接触域随机化及辅助全部显式关闭，precision保留开启；静态/运行时仍核对其确实未实例化。100步对应每环境2秒、2000次物理步，总计200万策略环境步及4000万物理环境步，PPO更新和正式课程预算消费均为0。保留35D/81D/6动作及原`[256,128,64]` ELU架构说明，但此入口不构建策略、rollout或optimizer，**不能据此宣称完整PPO 20k容量通过**。

轮输出cap分支明确标为`software_cap_only_not_identified`，不会将4.5再次乘15.8。它没有辨识的高速力矩曲线，仅供此容量探针；不能迁移为正式高速或跳跃驱动包络。原V4严格validator、ratio11历史身份和motor-side曲线转换保留。新loader只为工程显式电机prior服务，拒绝同时夹带motor-side曲线、重复减速比换算或改成训练发布。

合同以已归档P0的35D/81D任务接口作为模板，没有加载旧checkpoint或继承P0/正式S1预算。旧height_workspace几何表被显式清空，防止其旧model_spec SHA误用于新资产；不因此宣称新高度域已验证。奖励沿用接口模板只用于scene step执行，不是新奖励定标结果。正常训练入口若漏写`--validate-scene 100`、写错步数或要求resume/transfer，在创建run目录前拒绝。

最终V6 manifest SHA已分别核对并填入合同与独立表：`9542c788afd6e1b7f0af5e396062f123e511f63579cd8807087068e82e401c5c`，再重算设计表SHA写回合同绑定。`manifest_status`为`frozen_asset_identity_engineering_probe`，只冻结资产身份，不表示G1或训练发布。资产再变化须更新成对事实及设计表SHA；控制文件改变同样须更新其SHA及逐项设计值，不能自动以候选配置覆盖设计表。

在冻结源码目录及正确Isaac运行环境中，静态检查命令为：

```bash
python scripts/train_chassis.py --contract contracts/v6_asset_capacity_probe_v1.json \
  --stage foundation --num-envs 20000 --seed 918 --preflight-only
```

主任务完成冻结和资源检查后才执行下列GPU场景命令；本文件只准备命令，未启动：

```bash
python scripts/train_chassis.py --contract contracts/v6_asset_capacity_probe_v1.json \
  --stage foundation --num-envs 20000 --seed 918 --device cuda:0 --research \
  --validate-scene 100 --updates 1 --max-runtime-seconds 3600 \
  --run-dir /home/kaiser/robot-rl-sim60/experiments/REPLACE_WITH_NEW_CAPACITY_RUN
```

`--updates 1`只满足旧CLI参数边界，不进入PPO分支；合同总更新数为0且工程门禁强制走scene-only分支。应保存运行成功/异常回执、20k场景构造耗时、有效仿真步数/吞吐、GPU显存与利用率、WSL/宿主内存峰值。通信、随机化、非零策略动作、PPO采样/优化/封存及评测资源各是后续独立探针，不能以本场景探针替代。

最终静态回执：Kaiser独立`entry_validation`副本上，`test_chassis_preflight.py`、`test_pair_identification.py`、`test_capacity_probe_contract.py`和既有`test_twenty_thousand_envs.py`合计**120通过**（1.82秒）。真实CLI `--preflight-only`返回`asset_verified=true`及`design_preflight.status=static_passed`，记录20,000环境、seed918、1ms/20ms、decimation20、腿160/2.5、轮P0.6和所有扰动模块关闭。此前资产manifest更新时静态入口确实拒绝旧SHA；成对更新后通过。未创建Isaac场景、未执行GPU或PPO。
