# v27 普通大阶静态合成候选

本候选只实现 recipe/design/worker 的来源合成，不生成正式可部署 campaign/lineage/migration，不进行原生仿真或训练。当前 speed 不变。真实 completed v26 speed→v27 terrain 迁移、父 PT/full-state、账户和课程时钟必须由独立边界准入认证；不能重新从 REC 入场。

旧 v24/v25/v26 recipe 逐字节保持。v27 speed domain 完整复用 v26；terrain、jump、stepjump 在不可变 v23 terrain source 上应用 `ordinary_large_stair.sampling_candidate`，再沿原 component 去重、永久静止 floor 和4096分配合成。

普通阶原生池新增0.10、8行（单50/100/150/200mm；双50+62.5、100+125、150+187.5、200+250mm），来自 surface_traversal 0.10→0.02、surface_height_joint 0.04→0.02。8小阶0.42及其他复习池保持；该份额是原生 terrain source 中的配额，不是整个 integrated batch 的10%。后续课程保留这些 terrain 行复习。

来源审计保留不可变源101行的真实 source_coverage。新行标为 `derived_row`，原 source_name 为 null，另存 derived_name/spec/terrain SHA及显式 transform pin；旧 terrain 行标明 source transform 和原始源份额，不将派生行假称为旧 v23 原生行。transform 的文件/SHA绑定普通阶核心。

8大阶使用普通 body_climb mode0，由策略执行，不发 JUMP/STEP 请求。SCUT只映射现有 normal pools 和8小阶 bindings，heading只接24普通映射行；新ordinary_body_climb池始终保留自己的命令，不接受 planar/height/sequence/heading 二次拥有者。

静态 `origin_actor_update=0/actor_allowance=15000` 只用于合法schema检查，明确记录 `schema_only_actual_admission_parameters_required`。它们不是未来真实入场clock或预算；实际边界必须按认证 terrain component 起点、剩余 actor allowance 重绑定，并在后续阶段继承，critic适配不得推进actor时钟。该候选不启用 default-off recovery audit，不改 reward、gate、PD、物理或环境数。

必要CPU验证入口：

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:scripts \
  /home/yukikaze/isaacsim60-venv/bin/python -B -m pytest -q \
  tests/test_integrated_terrain_authoring.py tests/test_integrated_sampling.py
```

测试仅在临时目录生成 recipe/design/worker，显式不生成 v27 campaign、lineage、migration。byte identity、整数lane、source覆盖、派生provenance、普通命令拥有者和canonical worker/design通过不等于 native 启动通过、动作成功或已qualified。

本次必要CPU验证实际通过：采样与新authoring 28项（27.67s）；补充三个旧版本全部 design/worker/recipe byte identity及cached-transform改字节拒绝4项（36.50s）；最终 provenance分类相关3域检查3项（6.09s）。未重复旧全链native或运行PPO。
