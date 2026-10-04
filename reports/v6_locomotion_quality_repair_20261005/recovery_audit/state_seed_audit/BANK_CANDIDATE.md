# 首批 RecoveryBirthSample-v1 data-only 候选

[recovery_birth_sample_bank_candidate_v1.json](recovery_birth_sample_bank_candidate_v1.json) 已实际构建，等待 root review；没有提交，也没有接入训练。8个样本都是 `native_prefix_002/event_snapshots.pt` 中已观测物理状态：成功 lane1右侧、lane2仰腹，各取 `motion_unlock`（easy）、`first_pure_rl_post_physics` 和 `blend_last_mixed_post_physics`（medium）、`blend_entry_pre_effort`（harder）。没有插值、投影、速度清零或创建未观测sample。难度标签是候选分组，不是动态验证结论。

CPU builder/validator 位于 `scripts/recovery_birth_sample_bank.py`，测试位于 `tests/test_recovery_birth_sample_bank.py`。builder先认证已审核的report/events/快照/effective contract四个固定SHA，再用 `torch.load(weights_only=True,map_location='cpu')`读快照；不导入Kit/Isaac，不调用restore。`validate_bank`本身仅用标准库，校验schema、帧、named关节/类型/单位、源event身份、false资格标志和内容checksum。带 `expected` 时逐项比较独立重建的认证数据；`load_bank`要求调用者提供外部固定文件SHA。候选自身可重算的checksum不能代替来源认证。

每个 `RecoveryBirthSample-v1` 包含：

- root-link姿态xyzw，位置已减去源origin；保留world朝向，只支持平移到新origin，不能静默旋转场景。
- root COM **世界坐标**线xyz/角xyz六维速度，直接来自快照COM字段；保留全部真实非零速度。
- 18关节名→position/velocity/type/unit，同时带实际native列顺序。连续/转动关节为rad/rad/s；两条spring slide为m/m/s。
- source run/lane/case/birth/event/snapshot index、physics/policy/reference tick、物理时刻与完整event记录SHA。
- 固定asset manifest/model spec、源effective/worker合同及控制数学SHA。control/mechanics SHA是明确命名的**源合同投影**，内容随bank保存，机械投影额外绑定七个封存资产文件与求解器配置；不是宣称新episode已满足相同控制或机械条件。
- 原软件phase/dwell/完成状态仅为 `software_state_provenance.restore=false`。新episode完成信用始终false，源运动解锁/接稳计数不能直接恢复。

源条件有实读证据：startup四个lane总质量均13.8859987259kg；动力随机化模块已实例化，但汇总实际enabled_fraction=0、base_mass_scale均值1和COM offset绝对最大值0；contact汇总有效摩擦min=max=.5、nominal_fraction=1。各sample还保留原 motor_strength6、spring_strength1、transport lane使能和delay关闭条件。这些是**源full-role观察**。逐body的实际mass/inertia/COM数组、每lane完整材料参数未导出，明确 `unknown_not_exported`；bank不补填nominal，也不规定新episode的随机动力学/材料。

bank和每个sample都保持 `dynamic_seed_replay_validated=false`、`exact_snapshot_resume=false`、`qualification=false`；bank另有 `candidate_only=true`、`automatic_training_read=false`。路径在reports，没有修改V8姿态库、core或正式合同；Physics warmstart仍未导出。后续原生出生接口应从sample写root/q/dq一次，重置软件并重新赚取ready、40个BLEND子步和sensor/truth各1s接稳，再给新lesson结果。源2/4整链成功仅为单种子四个stand case的来源证据。

已执行 **27项CPU测试通过、零skip**：全部8个样本的root/COMworld6/q18/dq18逐值对照原始快照；标准库validator无torch/Isaac导入；四个来源SHA篡改在反序列化前失败；外部bank SHA、joint映射/type/unit、velocity frame、event篡改/重复、NaN/Inf、假软件restore/完成信用/资格均拒绝。重签checksum后的状态/event篡改仍不能通过来源对照。CLI重建及来源校验通过；[validation.json](validation.json)记录文件SHA和测试回执。

构建（默认输出本候选reports路径）或认证校验：

```sh
CUDA_VISIBLE_DEVICES='' /home/yukikaze/isaacsim60-venv/bin/python scripts/recovery_birth_sample_bank.py
CUDA_VISIBLE_DEVICES='' /home/yukikaze/isaacsim60-venv/bin/python scripts/recovery_birth_sample_bank.py --validate reports/v6_locomotion_quality_repair_20261005/recovery_audit/state_seed_audit/recovery_birth_sample_bank_candidate_v1.json
CUDA_VISIBLE_DEVICES='' /home/yukikaze/isaacsim60-venv/bin/python -m pytest tests/test_recovery_birth_sample_bank.py -q
```
