# 独立 V6 / 232 mm / 名义 105° 止挡研究候选

本候选从已冻结 V6 final_r3 生成，用户于 2026-09-29 确认 105° 同时为标定姿态内角和名义机械上限；实际有效止挡可能因误差达到106/107°。只将物理膝内角上限从110°改为105°，40°下限保留。

完整闭链、坐标轴、RR_link2 0.084 kg、所有19个刚体质量/惯量/质心、网格、零位和气簧232 mm全伸/80 mm行程均保持不变。源CAD和V5 reference_spec保留历史原貌。`parent_comparison.json` 要求仅两膝的两个范围端点存在物理差异。

- `model_kind` / `coordinate_schema` 保持V6 runtime兼容；用 `asset_revision` 与 manifest SHA 区分本候选。
- 标准输入 `inputs/mechanical_limits.json` 是schema2。默认物理上限固定105；105–107仅是独立机制研究的不确定域，不自动随机化。
- 同一次止挡设零导致有效止挡与encoder bias相关；不要独立随机两份同一误差。
- 名义 q=0 的平面膝内角为44.936658953811°，故左上限1.048303061004253 rad、右下限−1.048303061004253 rad。不是主动膝电机角105°。
- 实际有效止挡、接触刚度、encoder零位及气簧/执行器硬件辨识仍为G1 pending；training_ready与hardware_deployment_ready均false。

## 生成与CPU封装

依赖原V6生成环境：NumPy/SciPy/trimesh/MuJoCo/pxr。生成器拒绝覆盖已有bundle。`--cpu-only`只能使用真实通过且hash匹配的CPU、MuJoCo和parent比较回执，PhysX保持pending；没有复用旧110°资产的PhysX成功记录。

```bash
python tools/build_v6_closedchain.py --output urdf
python urdf/tools/validate_v6_asset.py urdf --output cpu_validation.json
python urdf/tools/validate_v6_dynamics.py urdf --steps 2000 --output dynamics_validation.json
python urdf/tools/compare_stop105_to_r3.py --candidate urdf --parent /path/to/final_r3 --output parent_comparison.json
python urdf/tools/seal_v6_asset.py --bundle urdf --cpu cpu_validation.json --dynamics dynamics_validation.json --comparison parent_comparison.json --cpu-only
```

CPU域验证为40–105°；动态检查是有限时长闭链/力方向/数值稳定性检查，不代表被动站立、训练任务、自由摩擦识别或硬件部署验收。新止挡的PhysX验证及完整地面接触检查未完成时必须显式保留pending。
