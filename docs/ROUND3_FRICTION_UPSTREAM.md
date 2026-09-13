# Round3 摩擦与地形：华南虎、复旦实际实现对照

核查日期：2026-09-14。依据本地上游源码及华南虎发布模型配置；没有重新训练上游模型。本文件细化B阶段设计，当前A的固定摩擦平地训练并未因此改变。

## 1. 实际启用配置

| 项目 | 华南虎 V14 普通Flat | 复旦 plane |
|---|---|---|
| 核查commit | `b8ff79f3df855faf9dc92f4a282bd80c42649466` | `8204e853dfd2ed06d85a322e1a998c3d20a3be2c` |
| 摩擦随机化 | 开启，轮body的startup事件 | 开启，创建环境时的shape回调 |
| 配置范围 | 轮静摩擦[0.5,1.2]，动摩擦[0.4,1.0] | shape的单一`friction`字段[0.6,1.4] |
| 桶数量 | 64个材质三元组桶 | 64个摩擦桶 |
| 静/动一致性 | `make_consistent=True`，把动摩擦裁到不高于静摩擦 | 本回调未独立采样静/动摩擦字段 |
| 分配粒度 | 所用Lab事件按env×shape分配桶，左右轮可以不同 | 每env选一个摩擦值，赋给该机器人所有shape |
| 采样时机 | startup分配；所查任务没有每回合重置或每步摩擦重采样 | 环境创建；reset路径没有再次调用该材质随机化 |
| 地面静/动摩擦 | 1.0 / 1.0 | 0.5 / 0.5 |
| 合成模式 | 仿真默认和地面均显式`multiply` | 所查Python配置未显式指定combine，不能凭shape值认定有效轮地值 |
| 默认地形 | plane；Rough另有独立任务 | plane；解析时强制关闭地形curriculum |

静、动摩擦的“动”描述接触滑动状态，并不表示该参数每帧变化。两家的主要机制都是同时训练一批具有不同物理参数的环境。

华南虎源码的旧“暂时关DR”注释不代表当前事件未启用。发布模型`flat_and_rotation/2026-07-29_23-20-03/params/env.yaml`也记录了上述轮材质范围、64桶、startup及地面multiply/1.0。

### 回弹系数也要按接触对理解

华南虎轮shape的restitution采样[0.02,0.2]，但所查地面restitution为0、combine为multiply；按这组配置组合后的轮地回弹为0，不能把轮材质范围当作轮地有效回弹范围。复旦plane配置的shape restitution为[0.6,1.0]、地面为0，但所查Python路径未显式给出combine，不能直接比较二者或推导出相同的有效回弹。

## 2. 地形与课程究竟开启了什么

### 华南虎

- `WheelbipeV14RoughEnvCfg`选择`RM_ROTATION_TERRAINS_CFG_99`，包含小台阶、低坡/反坡、阶梯坡/反向阶梯坡、平地和随机起伏。
- Rough运行配置切换为地形generator，并改用地面相对高度；不能沿用平地的世界z高度定义。
- generator的`curriculum=True`按difficulty生成地形行；同时env的`curriculum=None`。**难度分层地形生成不等于已启用按策略表现自动晋级。**
- 发布的rough训练快照从flat的`model_8000.pt`初始化，并记录`resume_training=false`，支持先平地后粗糙地形的分阶段迁移思路。

### 复旦

- plane配置虽然写`terrain.curriculum=True`，但`mesh_type='plane'`，`_parse_cfg()`将课程设为False；它不是已启用的粗糙地形训练。
- 创建heightfield/trimesh和地形晋级函数存在，不代表plane运行时走过这些路径。
- plane的物理DR开启，但`push_robots=False`、`randomize_action_delay=False`。jump属于另一任务，不能合并成plane默认能力。

## 3. 对应我们的B阶段

采用两家共有的**startup分环境随机化**作为首个可复现对照：

1. 生成64个`(mu_static, mu_dynamic, restitution)`材质桶，初始化时分配，回合reset不重新分配摩擦。保存桶表、env映射及实际PhysX shape读回，精确resume时恢复映射，不能只靠相同seed声称物理分布未变。
2. 约30%环境固定名义材质，其余分配随机桶。名义环境另用mask记录，不要求随机桶恰好抽到0.5。
3. 初期每env两轮共用一组材质，便于测量统一地面的起停和附着。这里借鉴复旦的env级一致性，但只改变轮shape；机身/腿的接触材质不随之一起改。
4. 左右轮不同材质作为后续独立对照或留出测试，参考华南虎按shape分配的做法；空间材质区域用于进一步检验驶入/驶出低摩擦区域，不能靠每步随机跳值模拟。

### 范围按“有效轮地摩擦”制定

| 分布 | 有效静摩擦候选 | 有效动摩擦候选 | 回弹 |
|---|---|---|---|
| 名义 | 0.5 | 0.5 | 0 |
| 初始DR | [0.5,0.7] | [0.4,0.6]，再裁到不高于所采静摩擦 | 0 |
| 扩展候选 | [0.4,0.9] | [0.3,0.8]，同样保证静/动一致性 | 先保持0，回弹另做单项辨识 |

先在有效系数空间采样静、动参数并做一致性处理，再转换为轮材质配置。保留当前地面0.5、`average`时：

`mu_wheel = 2 * mu_effective - 0.5`

因此初期wheel静摩擦范围为[0.5,0.9]、动摩擦原采样范围为[0.3,0.7]；扩展候选分别为[0.3,1.3]和[0.1,1.1]。这些值与华南虎地面1.0、multiply下的轮材质数字不能直接横比。静/动裁剪后边缘分布不再是两组独立均匀分布，报告应保留实际桶表。

上述范围是围绕V40已验证名义材质的研究候选，不是宣称两家使用了这些范围，也不代表实机地面已经辨识。B阶段的实现、材质读回、摩擦与滑移评估尚待完成；当前A仍是显式0.5/0.5平地。

## 4. 分布变化与不同地形分开验收

先验证多摩擦平地；再从已验证平地策略迁移到缓坡、小起伏及小台阶。地形阶段应按V40的60mm轮半径、真实膝界和碰撞净空确定尺度，不能照搬上游地形高度。引入地形时统一命令/reward/日志的地面相对高度reference。

评价至少包含固定摩擦各档、初始条件/测试seed留出、纯直行/制动/旋转、左右轮差异，以及空间材质过渡。当前最低圆柱点proxy只能辅助诊断，真实滑移仍需轮地接触身份/支持接触条件与接触点速度；不同摩擦的平地成功不能代替不同几何地形的成功。

## 5. 可复查源码

- [华南虎轮材质事件及范围](https://github.com/scutrobotlab/wheeled-legged_RL/blob/b8ff79f3df855faf9dc92f4a282bd80c42649466/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_V14/env_cfg.py#L128-L170)
- [华南虎继承的仿真/地面multiply配置](https://github.com/scutrobotlab/wheeled-legged_RL/blob/b8ff79f3df855faf9dc92f4a282bd80c42649466/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe25_v3/env_cfg.py#L255-L278)
- [华南虎发布Flat模型材质快照](https://github.com/scutrobotlab/wheeled-legged_RL/blob/b8ff79f3df855faf9dc92f4a282bd80c42649466/pretrained/26_infantry/flat_and_rotation/2026-07-29_23-20-03/params/env.yaml#L247-L284)
- [Lab v2.3材质桶采样、静动裁剪及shape分配](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab/isaaclab/envs/mdp/events.py#L220-L278)；华南虎快照函数标识为`isaaclab.envs.mdp.events:randomize_rigid_body_material`。这是所查Lab实现，不是仅凭华南虎git commit证明其所有发布run使用了这个SDK版本。
- [华南虎Rough任务选择](https://github.com/scutrobotlab/wheeled-legged_RL/blob/b8ff79f3df855faf9dc92f4a282bd80c42649466/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_V14/env_cfg.py#L1675-L1757)
- [华南虎Rough运行覆盖及课程](https://github.com/scutrobotlab/wheeled-legged_RL/blob/b8ff79f3df855faf9dc92f4a282bd80c42649466/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_V14/cfg_utils.py#L567-L624)
- [华南虎Rough地形组成](https://github.com/scutrobotlab/wheeled-legged_RL/blob/b8ff79f3df855faf9dc92f4a282bd80c42649466/source/agent_tasks/agent_tasks/manager/mdp/isaaclab/terrains.py#L672-L698)
- [复旦plane摩擦/回弹/推扰/延迟配置](https://github.com/yly-true/fudan_rl_wheel_leg/blob/8204e853dfd2ed06d85a322e1a998c3d20a3be2c/plane/wheel_legged_gym/envs/base/legged_robot_config.py#L158-L181)
- [复旦创建环境时的64桶分配](https://github.com/yly-true/fudan_rl_wheel_leg/blob/8204e853dfd2ed06d85a322e1a998c3d20a3be2c/plane/wheel_legged_gym/envs/base/legged_robot.py#L438-L484)
- [复旦地面参数写入](https://github.com/yly-true/fudan_rl_wheel_leg/blob/8204e853dfd2ed06d85a322e1a998c3d20a3be2c/plane/wheel_legged_gym/envs/base/legged_robot.py#L1297-L1304)与[plane关闭地形课程](https://github.com/yly-true/fudan_rl_wheel_leg/blob/8204e853dfd2ed06d85a322e1a998c3d20a3be2c/plane/wheel_legged_gym/envs/base/legged_robot.py#L1597-L1605)。
