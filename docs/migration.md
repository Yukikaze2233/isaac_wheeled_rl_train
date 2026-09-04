# 自有机器人迁移清单

把本框架从参考模型迁到自有轮腿机器人的完整清单,按依赖排序。
配套说明:训练仓 README「资产」节 + 部署仓 CONTRACT.md。

## Step 1 — URDF → USD(最耗时,慢慢做)

- [ ] SolidWorks 导出 URDF(质心/惯量张量逐项核对,勿用自动估算)
- [ ] Isaac Sim URDF Importer:root=base_link、**不合并 fixed joint**、自碰撞关、
      惯量保留 URDF 原值
- [ ] 导出 USD 至 `$WHEELED_RL_ASSETS_DIR/wheeled_biped/wheeled_biped.usd`
- [ ] USD viewer 检查:默认站姿、质心、碰撞体、关节轴

### 关节命名约定(env 正则依赖,不改名就要同步改 env)

```text
主动腿关节:  {left,right}_front1_joint / {left,right}_rear1_joint
被动闭链:    *_front2/3/4_joint, *_rear2_joint, *_spring1_joint   (并联腿才有)
气弹簧:      *_spring2_joint (prismatic)                          (有气弹簧才有)
轮:          *_wheel_joint
```

串联腿机器人:直接删 legs_inact/spring 组与 no_fork 类奖励,迁移难度显著更低。

## Step 2 — 资产配置(`wheeled_world/assets/__init__.py`)

- [ ] init_state:各关节默认角(= CONTRACT 的 default_dof_pos)、spawn 高度
- [ ] 执行器组:每电机 Kp/Kd(**真机下发值**)、力矩/速度限幅、减速比
- [ ] armature = 转子惯量 × 减速比²(逐电机查手册;漏掉是高频抖动的头号来源)
- [ ] 串联腿:删 legs_inact / spring 组;有实测曲线 → 用 `actuators/m3508_curve.py`
      加载 CSV(比常数 effort_limit 更真实,尤其跳跃等大负载工况)

## Step 3 — 环境配置(`env_cfg.py` 新建 `OurRobotFlatEnvCfg`)

- [ ] 关节行程 → leg_action_scale(默认 0.5 是 ±0.5 rad,按实际行程缩放)
- [ ] `height_range` / `default_height_cmd`(按车体几何)
- [ ] `max_wheel_vel` / 轮力矩上限(用电机曲线模型时由曲线限幅,常数只做兜底)
- [ ] 并联专属项(串联腿):删 no_fork、五连杆 links_length/alpha_offset
- [ ] 有气弹簧:按实测曲线改 `spring_settings`(力值/行程/偏移)
- [ ] 频率不动:200 Hz × 4 = 50 Hz 已验证,不要动

## Step 4 — 域随机化对表(events.py 调用点)

原则:**每一项都要有真机依据,先窄后宽**。数值意义随轮径/质量/重心完全改变,禁止盲抄。

| 项 | 依据来源 |
|---|---|
| 质量/COM 缩放 | 装配公差 + 电池 SOC 重量变化实测 |
| 轮摩擦 | 真机打滑阈值实测(不同地面各测) |
| 关节摩擦 | real2sim 辨识(见下) |
| PD 增益缩放 | 下位机增益离散化误差 |
| 延迟范围 | 实机回环延迟实测(见部署 docs/sim2real.md) |

**real2sim 辨识流程**:真机架空跟踪预设轨迹录 bag → MuJoCo 同轨迹回放 →
对比关节跟踪曲线 → 调仿真摩擦/阻尼直至重合。轮电机可直接辨识静摩擦+粘滞阻尼;
关节电机单关节辨识意义不大(误差主要来自结构耦合),用整体曲线拟合。

## Step 5 — 训练(服务器)

```bash
export WHEELED_RL_ASSETS_DIR=/path/to/usd
./isaaclab.sh -p scripts/train.py --task OurRobot-Flat-v0 \
    --num_envs 4096 --max_iterations 20000 --headless
```

- 先窄 DR 训通(20k iter),再逐项放宽续训
- flat 达标 → 加延迟随机化 → Rough(`WheeledBiped-Rough-v0` 模式)→ 特殊模式
- 每轮改动从 checkpoint 续训(闭环 ~1h 级)
- 算力参考:4090 云卡,flat+rough 全程 300–500 卡时

## Step 6 — Sim2sim → 真机

1. `export_onnx.py` 导出 → `check_onnx_contract.py` 校验
2. 部署仓 MJCF 换成自有模型(关节名同合同)→ `mujoco_sim2sim.py` 曲线对齐
3. 逐项过部署仓 `docs/sim2real.md` 清单,再上真机
4. 真机首轮:低速、站姿附近,observation debug topic 逐帧核对 35D

## 常见坑(按踩中概率排序)

1. 投影重力方向/符号与训练不一致 → 实机前倾后仰颠倒
2. 动作 scale 忘改 → 腿目标出关节限位,瞬间打满
3. armature 漏算 → 腿高频抖动
4. 延迟随机化不足 → 实机抖动,仿真看不出来
5. default_dof_pos 训练/部署不一致 → PREPARE 后姿态偏移
