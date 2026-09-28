# stop105 候选交付回执

2026-09-29 生成并封装独立候选 `v6_232mm_stop105_nominal_v1`。用户确认105°是标定姿态和名义机械上限；实际106/107°作为未分离零偏/止挡误差的研究不确定性，不自动随机化。

当前完整验证包为 `urdf/`；最初 CPU-only 包完整保留在 `urdf_cpu_only_v1/`，没有改写其文件或哈希。原 `model/纯底盘_v6_232mm/urdf` final_r3 和原20k合同未改。

| 身份 | SHA-256 |
|---|---|
| 当前 manifest | `875fa71e89d4d669af0a5471b2334f878d302e06f1b354fc245b95cb5d6f29ba` |
| CPU-only manifest | `1043f540d81247a0c7088467ea63b9849a422dc96742ebf062f583202f7521b7` |
| 父 final_r3 manifest | `9542c788afd6e1b7f0af5e396062f123e511f63579cd8807087068e82e401c5c` |
| MJCF | `770340546b0d6fabdb1dba87dfa03ecfaacaea20a3b5b8de5c1984fb377dff3f` |
| model_spec | `7429f6e59369e590e0fd96cba8b45fd60c362a0231c4bcc2e80a4c4688eadd88` |
| USD | `f9a1e4a9350ff0d98af187ed287f73428a10ed339c9ef7b9de04688fa30f0f98` |

Kaiser 当前包：`/home/kaiser/robot-rl-sim60/experiments/v6-stop105-build-20260929/physx_v1`。CPU-only保留在同实验目录的 `model/urdf`。构建源档、CPU/动力学回执和 PhysX log/report 也在该独立实验目录；未安装新依赖。

## 变化与证据

严格对照 `parent_comparison.json`：只改变 `L_joint2.upper` 和 `R_jonit2.lower`，从±1.135569523603969变为±1.048303061004253 rad，对应膝内角110°→105°。替换这两个字段后整个spec精确相等；URDF、MJCF、USD也只相应范围不同。

所有19body质量/惯量/COM、RR_link2=0.084 kg、total=13.886 kg、网格文件、闭链约束、轴系/原点、nominal、232 mm全伸/80 mm行程、气簧名义曲线、主动/被动阻尼都与父版本保持精确一致。没有引入执行器拟合参数或更改encoder offset。

新版 builder 使用schema2的机械来源输入；保留runtime认识的V6 model_kind/坐标schema，用独立asset_revision与父SHA区分。CPU validator新增URDF、compiled MJCF与USD角限位的一致性检查。CPU-only sealer要求真实成功且hash匹配的CPU、动力学、parent对照回执，拒绝继承PhysX成功状态。

## 验证

- 封装84个文件在本地逐项SHA复核；本地再次与未变的final_r3执行严格物理差异比较。
- CPU 40–105°域75姿态，通过6闭链/rank12/FK/气簧虚功/坐标对照；max closure `3.414e-14 m`。源CAD及原reference文件作为历史来源保留。
- MuJoCo 3.8.0：三个各2000步、dt=0.0005 s的自由基座有/无气簧及小力矩场景通过；关闭闭链600步negative control按预期明显散架。有效闭链最大gap `0.20925 mm`，无MuJoCo warnings。软止挡最大穿透 `0.008322 rad≈0.477°`，这个“通过”仅指既定有界数值/闭链检查，不等于硬件角限位精度验收。
- PhysX：1env、固定底盘、1 ms、2°小摆、3.02 s，6真实闭链；max gap `7.83034 μm`、销距/滑轴差 `0.107646 μm`。reset后没有被动pose写入、没有逐帧IK。启动检查实际PhysX knee/slider范围与105 spec一致。小摆未触及上止挡，不证明边界冲击或真实止挡刚度正确。
- CPU-only封装工具50项离线测试通过，包括错误hash、缺/失败回执、遗留PhysX、非法readiness、重复封装和`python -O`。失败路径不改bundle。

## 保留边界

`training_ready=false`、`hardware_deployment_ready=false`，`G1_status=pending_hardware_identification_and_effective_stop_zero_separation`。完整地面接触、止挡接触/刚度、实车坐标/力矩/气簧温度老化辨识和任务域验收仍待完成。本次没有运行20k测试。

资产限位来源得到修正，并不说明候选motor plant已经通过。若回放RMSE改善依赖大量模拟止挡接触，不能将这种改善当作自由运动电机模型验证成功；本候选不吸收任何拟合出的motor参数。
