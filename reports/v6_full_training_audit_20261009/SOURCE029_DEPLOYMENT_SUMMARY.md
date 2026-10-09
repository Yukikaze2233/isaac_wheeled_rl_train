# source029 同 v26 串行接续部署摘要

最终新源码已完成stage003、CPU预检、一次origin更正、唯一原wrapper启动及实际speed首actor恢复核验。2026-10-09 05:20（UTC+8）`first_actor029_001.json` status=passed。本文编辑不执行SSH、测试或训练，仅据root保存回执与冻结树独立核读；核验通过不授新能力或机械/实车资格。

| 环节 | 已核事实 | 范围 |
|---|---|---|
| 错误候选拒绝 | `55ead9e3ce30af09fede479637fc9c032960dbb4` 非真实source028父树，带额外合同和env.py变更；全合同比较拒绝 | preflight未进入origin/launch，0 PPO |
| 错误暂存隔离 | 四件源码目录、tar、source_revision、stage目录rename至 `source029_staging_rejected_archive001/`，原字节和intent/receipt保留 | 原job/claim/consumption/state/REC PT及source028回执SHA保持；0 PPO |
| 最终冻结 | `e5bd9c1367b875d7f229b9a1930647013d023372` 的直接parent为 `335542ddc753b84b2778609b92803064682198a9` | 八个owned文件闭集；276合同和4305非owned tracked文件blob/git mode不变 |
| 真实双目录CPU测试 | 1项通过99.81s；另有targeted lineage/origin/dispatch 46项通过153.09s | 生命周期准入，不是仿真行为或PPO收益 |
| stage002权限拒绝 | 4310共同文件字节相同，4306×0644→0664及4×0755→0775；原因git默认tar权限掩码 | origin检查拒绝，0 PPO、无更正/launch |
| 第二次失败暂存隔离 | 四件对象及preflight/origin日志原字节保存在 `source029_staging_rejected_archive002/` | 原job/账本/REC PT等SHA保持，不放宽mode判据 |
| 最终stage003 | 05:10:28 exit0，同一e5bd9提交显式 `tar.umask=0022` 重tar，4313文件，archive `7cb0a586a1c83f49325a244d3fe8793c12117ac3bedf69facafd9d06bbc57bb0` | 0 native scene、0 PPO、qualification=null |
| 真实preflight003 | exit0，actual dispatch=speed/terrain/jump/stepjump，origin check-only通过 | 0新模拟场景、0 PPO |
| 真实correct003 | exit0，一次origin更正，父REC41393/SHA e04c…3967与全部19次消费保留 | 更正工具0 PPO，不改合同或参数 |
| 真实launch003 | 05:12:11提交，wrapper4141407→supervisor4141412→speed worker4141576 | 唯一owner，实际speed/attempt003；未重复REC |
| 当前worker原生启动 | runtime_configuration_passed，4096/50Hz策略/200Hz物理PD | 新父实际恢复另由首actor证明，不借旧REC14169工程预检授功能 |
| 首actor核验 | passed，9actor/8critic/17Adam初态exact；首actor41404、task/joint actor1、完整谱系27657，九actor张量改变 | 10次critic计费；读者0 PPO/0信号/0 GPU，qualification=null |

CPU测试使用synthetic模型SHA/时钟fixture，真实old/new绝对目录与归档；覆盖旧origin不能掩盖新增消费、correct_origin、open_account幂等、latest speed父来源、新目录train命令、实际worker admission、四域串行派发，以及wrapper实际resume024留存job/receipt配对。错误留存job与合同改动拒绝。此前同一个fixture源码目录的三项helper CPU测试不能替代这个跨prefix覆盖。

八文件为：`scripts/run_task_campaign.py`、`scripts/chassis_remote_job.py`、`scripts/repair_known_integrated_origin.py`、`src/wheeled_tasks/chassis/integrated_lineage.py`、`src/wheeled_tasks/chassis/integrated_production.py`、两份origin/early-production测试和source029设计文档。机械模型、`env.py`、所有reward/课程合同、物理PD、执行器响应、DR与预算保持。

原helper SHA仍为 `3864f047ffea30989cce30db1bd697c20d5f217d70286d09c77c12198c05b963`；最终stage003 source-revision SHA为 `ec593b9ce6f6c7a80cf3cdbd78131f7fd29f42436b5b48f68f390df2ee059b62`。stage、preflight、correct、launch的产物分别独占发布，不覆写、不自动重试一次性intent。旧错误候选归档SHA `abdedd3f6232d7eb10cc8691fdf9a053e399b43288c7b30af2a5fdb29b9dc840` 与旧revision SHA `7e7195fa7710f195ef61ba3b7a356d04402773140f1b612732be5ff9c2989121` 保留。stage002的归档 `55e3af79ede31160e0cf2f6993cc535bf384ef0ad0e52dbfbe31ccbad3518423` 与revision `f825f71278babaeba59eaed81c2c2ef545386f2a33bc0bba80e3b320bc5156dd` 在archive002保留，未替换其失败证据。

实际入场的真实父为 `train/recovery/attempt_010/train/model_final.pt`，SHA `e04c2f7a20c462b156bd6e21c4eaf2d14bec5e2a2a7956e327c26e383d603967`，累计消费41393/actor消费41223/REC14188/完整谱系27646。误派发的19个actor全部保留，未回退41374或退款。实际首域为speed，先10次计费critic，再actor；同模型后续顺序terrain→jump→stepjump，保留冻结复习配方。按旧speed消费5093计算，本域最多新增14907次更新＝10critic＋14897actor；不得把预算端点写成已达质量资格。

首critic封存41394时actor/std、stage/task actor与component课程时钟保持，critic发生学习。首actor封存41404/SHA `012dc8be8be7e98600ffa7c9aced166cb4eec6565fd25d6a9c84a98c45eb8afd` 的actor消费41224、stage27557、task/joint1、完整谱系27657，九actor张量（含std）改变。中间critic第2–10次没有逐次immutable张量封存，仅由phase/时钟记录证明计费；不称逐次tensor核验。父 RNG 保存结构及对应 seal 通过准入认证；CPU 认证没有安装并读回 CPU/CUDA RNG，新episode重置后的RNG也不与原环境逐位比较。此范围不影响已有actor/critic/Adam状态核验。

核验窗口截至05:19:56（UTC+8）：successful13=10critic＋3actor、speed task3、stage27559；authoritative account消费41406/actor41226/剩79594，speed历史消费5093＋新13＝5106。reader开始05:19:50、结束05:20:08，进度与账本独立只读采样，非原子runtime快照。只证明新的同actor speed正式学习已经进行，不证明高速姿态/驻停精度改善。

现有ONNX交付及短回放结论保持；启动与恢复核验不等于学成高速、跳跃或台阶。新模型效果待完整训练窗与同条件评估后另计。

证据：[真实父树对照](source029_true_base_verification002.json)、[错误基底隔离](source029_quarantine001.stdout)、[权限拒绝诊断](source029_mode_rejection002.json)、[权限失败与CPU日志隔离](source029_quarantine002.stdout)、[最终stage003](source029_stage003.stdout)、[preflight003](source029_preflight003.stdout)、[correct003](source029_correct003.stdout)、[launch003](source029_launch003.stdout)、[首actor恢复passed](first_actor029_001.json)、[跨目录独立审查](SOURCE029_PREFIX_ALIAS_REVIEW.md)。
