# Terrain v14 冻结源只读预审

结论：terrain v14 完整继承 speed v14 的控制、通信/观测延迟、轮响应、35D/81D ABI、机械边界和本次两项普通运动奖励修复。terrain 自身的旧技能、评测定义和父树保留。冻结设计物化及 campaign **30 个角色** CPU preflight 通过；所检查的普通静止/动态奖励相位边界没有复现旧被动落地正奖励或姿态成本相位逃逸。此结论限于合同、调用点及函数级 CPU 检查，不代表原生物理质量、地形能力或实车验收通过。

依据是实际部署冻结源 `b4ce80964d47c0127e0b3d9929ad35a620bbf204`、`contracts/v6_task_campaign_v14.json` 以及绑定 worker/design 的 SHA。没有使用 AGENTS 的历史 v10 状态作为现状，也没有修改合同、运行源码或远端进程。CPU runner 从 `git show b4ce809:...` 加载项目模块，因此没有加载当前用户修改的 `preflight.py`；差异哈希保存在 `source_receipt.json`。

## 继承和有意差异

| 项目 | Terrain v14 |
| --- | --- |
| 仿真/PD与策略 | 200 Hz / 50 Hz，`.005/.02 s`，solver `64/32`，history=1，35D actor / 81D critic / 6 action；与 speed 完全一致 |
| 控制/机械 | 同资产 manifest、control math SHA、六轴顺序、160/2.5 PD、气弹簧、工作裕量、闭链/膝/气弹簧硬终止和命令联合包络；硬件可部署仍为 false |
| command transport | enabled=true；旧 `usb_transport=false` 为被替代旧模块，不表示运输关闭 |
| signal delay | train enabled=true，episode enabled fraction=.75；三档 obs/action 上限 `20/15、40/30、80/60 ms`，比例 `.60/.35/.05`，每策略 tick `±5 ms` jitter；PD 使用 fresh physics feedback |
| response | 同轮侧已校准 armature/friction/damping、同 response evidence SHA；四腿仍为名义研究先验，未接入未 qualified 腿 bag |
| 普通奖励修复 | stationary L1 `weight=-2, |vx command|<.1`；pitch/roll coefficient floor `3/4`；dense tracking、precision、reference、finite landing 等均与 speed 相同 |
| 非标准出生/松杆教学 | train 开启，零力矩 hold `.2 s`、capture `1.5 s`；5% release-stop；fixed suite 均关闭这两项教学 |
| Speed 独有课程 | terrain 没有 `locomotion_difficulty` 的 2→5 m/s 高速渐进；terrain release-stop 平移范围 `.3–1.5 m/s`、yaw `.5–2 rad/s`，speed 平移 `.5–5 m/s`、yaw `1–6 rad/s`。这是冻结的有意差异，不是核心继承丢失 |

合同证据：terrain worker `:9` 时钟/ABI、`:14682` command slew、`:14748` finite landing、`:14862` command transport、`:14920` response 范围、`:14987` L1、`:14993` posture floor、`:15001` response、`:15047` evidence、`:16798` signal delay、`:16848` nonstandard start、`:16895` release stop。`static_diff.json` 保存 47 项核心字段的相等比较及完整旧 terrain v6→v14 字段差异。`working_margin_weight=0` 与 speed 相同；不能将其写成已启用的软成本，硬机械终止仍在。

## 采样、普通上阶和保护

旧 terrain **37 个 scene group 原技能 spec 逐项不变**；原比例统一乘 `.95`，新增六个 release-stop group，共 43 组、4096 人口，无零人口组。terrain 没有复制 speed 的全部 135 组行为库存。

| 池 | 名义比例 | CPU 实际人口 |
| --- | ---: | ---: |
| 普通上/下台阶等 12 技能 | 57% | 2338 |
| slope up/down、cross-slope、rough、material | 19% | 780；rough=156 |
| 九固定高度静站 230…430 mm | 8.55% | 351；每档=39 |
| 连续高度静站 | 5.70% | 233 |
| 平地 entry/exit、静态/连续高度、正反 yaw | 4.75% | 190；10 组每组=19 |
| 三种 release-stop × 两种高度 | 5% | 204；每组=34 |

上阶包含 rolling 30/60/100 mm、3×40 mm，以及 body-climb 150/200/250 mm、双级 rise 200+250 mm（顶面 200/450 mm，踏面 800/600 mm）。这些普通 rolling/body-climb 的 `kind` **不进入** `skill_commands.py:193` 的手动 step reference 映射，因此 `reference.terrain_mode=0`，不会借手动 step 相位豁免新 posture floor。`step_assist`/`jump_assist` 均关闭；body-climb 几何只用于奖励与验收，没有新增 actor 真值输入。

Body-climb 高度放宽仅在当前 riser 几何区域，经过/完成后恢复；只缩放高度项，pitch/roll、速度误差、努力和硬机械检查保留（`step_climb.py:169`，`rewards.py:151`，`env.py:1521`）。CPU 检查 scale=`1→0→1`，已完成路线即便重新处于边缘也为 1；持续留台 80 tick 只产生一次 platform event，持续无支撑达到 `.08 s` 后 success 无效（`step_climb.py:143`）。合法机身顶面支撑允许普通攀爬，但 riser 撞击/非法接触和姿态/机械检查保留（`step_climb.py:96`，`env.py:1373`）。

旧 terrain evaluation 定义完全保留，包含广域平地/rough/普通上阶条件；`required_case_names` 与对应设计经过 preflight。但原生自动调度不进行这些门的验收：assessment interval=0，endpoint 自动前进要求 evaluation=false。采样保留和配置预检不能保证不遗忘；最终仍需另行完整窗口/固定评测。fixed suite 的 **signal_delay=false**，其成绩本身也不验收训练随机延迟的尾部鲁棒性。

## 普通奖励相位检查和边界

`phase_cpu_checks.json` 使用冻结奖励函数及普通调用链，检查 stand、forward、ordinary yaw、release-zero 在 GROUND/LANDING/RECOVERY。对同一物理量，当前 `env.py:1393` 在普通 LANDING/RECOVERY 恢复高度 support，三种 phase 的所测奖励项一致；普通 landing cost 权重在 100 tick 接触/相位切换中始终为 0（`env.py:1464`，`rewards.py:34`）。显式 drop 的 `.6 s` 成本窗口不因接触抖动重启，且只支付零中心非正成本。

加速/减速 `±1.5 m/s²`、不同 elapsed 并不改变普通 posture target，也不豁免 floor；普通 yaw 仍支付低 vx 的平面速度 L1。失去 support 时新增 pitch floor 仍为负。普通 dense stationary 的 quiet blend 连续；同一可见高度命令没有隐藏 h-dot 奖励目标（`rewards.py:169`、`:189`、`:210`、`:437`）。

手动 step 的豁免是单独的有限命令参考，不是无限物理 RECOVERY：CPU 5 s hold 示例在 5.5 s 自动回到 terrain_mode=0（`references.py:78`、`:155`）。普通下阶的已有 manual reference 语义保留；不能把本次普通上阶检查扩展为所有手动跃迁已通过。FLIGHT 时部分 support 门控的高度/驻停项及原生接触/终止状态机未在此 CPU harness 运行，故不作全相位激励最优性的证明。没有确证需要立即修改本运行合同的问题。

## 自动终点和资格

terrain 从自己封存的 v6 terrain 学习状态、actor **1186** 恢复至 **2186**，新增 1000 actor；10 critic-only 单独计费，不推进 actor。父树仍为 flat，**不是从当前 speed v14 checkpoint 转接权重**。证据：campaign `:54`、terrain worker `:16787`、`nonstandard_start_revision.py:116`、runner `:584`/`:631`/`:647`。源码同时认证自身源 SHA、保留 actor/critic/Adam 和消费、不退款。

该 1000 actor 窗在 nonstandard relative update=1000 达到概率 nominal/gentle/wide=`.70/.20/.10`；计划的 relative 1500 末档 `.65/.20/.15` 不在 terrain 本次 endpoint 范围内。这是教学覆盖边界，不是已有资格。自动封存结果仍为 `unassessed_research_candidate`，`domain_qualified=null`，没有 accepted/deployment 模型。

复现（纯 CPU，无 SSH/Isaac/GPU）：

```sh
CUDA_VISIBLE_DEVICES='' /home/yukikaze/isaacsim60-venv/bin/python reports/v6_locomotion_quality_repair_20261005/terrain_audit/audit_terrain.py
```

输出 `static_diff.json`、`cpu_preflight.json`、`phase_cpu_checks.json`、`source_receipt.json`。本次进程 CUDA initialized=false，Isaac 未导入，SSH/仿真调用均为 0。这些文件不改变运行合同或部署。
