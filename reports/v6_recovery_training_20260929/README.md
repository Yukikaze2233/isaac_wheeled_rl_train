# V6 恢复课程：部署与验证回执

截至2026-09-29 18:24（UTC+8），Kaiser任务`v6-new-asset-20260929T090222Z-7aef2a`已完成四轮迁移评测并进入S1正式PPO更新。冻结代码`3051c6d6b8da3dd6fa71d5c0200b7884e0da7879`，无overlay；实读新增2次、累计消费380、学习谱系220。后续进度须读远端，不能把此快照当实时训练计数。

- [正式PPO启动核验](ppo_start_observation.json)：20000环境，首两次actor/critic更新完成，每次480000转移；首轮约44秒。四轮迁移评测完成，原有34项双seed通过成绩全部保留；新增54项中30项双seed通过、24项未通过，不能将其记为本轮PPO进步。18:23首轮采样时GPU实读92%，18:24更新期间采样90%，显存13519 MiB；瞬时利用率随采样/优化阶段波动。此前低占用为建场期，不把worker进程启动或监督器training标签当作PPO更新完成。

- [启动参数与归档](launch.json)、[部署进程/阶段核验](deployment_observation.json)：旧/新合同同checkpoint、双seed评测；之后配置20000环境，总消费上限8500，50Hz策略/1000Hz物理及PD，网络35/81不变。
- [父状态选择](v6_kaiser_parent_selection_20260929.json)、[旧运行封存](source_run_sealed.json)：恢复218学习谱系的完整actor/critic/Adam，消费从378接续，不退款。
- [LR决定](resume_learning_rate_decision.json)：首个PPO worker从恢复的Adam LR半减到约0.0001950922，之后继承原生自适应LR。worker resume_verification.json已确认actor/critic/optimizer恢复一致，且该LR覆盖实际生效，详见启动核验回执。
- [CPU验证](cpu_validation.json)、[40角色合同预检](cpu_preflight_40_roles.json)、[最终代码复核](release_code_review.json)：控制/合同313项与监督器104项测试集通过，计数不可与早前重叠测试直接相加。
- [最终20k GPU探针](kaiser_20k_final_probe.json)：实际60个1ms物理步、每策略步20次新反馈PD，18894个RL区间和1106个脚本区间正确分离。只有3个零动作策略区间，不代表PPO更新或持续吞吐。
- [最终108次工程回放](probe07_summary.json)：65/108成功，两次均过31/54；平地61/68、台阶4/40。14次交接前失败排除PPO，27次提前解锁及2次actor失稳保留actor终止负反馈且禁止bootstrap。源码逐文件匹配3051c6d。
- [未完成的教学环节](trainability_review.md)：脚本失败无PPO改善入口、平台保持/重返缺稠密过程教学及可观测任务定义、阶段前脚本可达性额外自动门尚未实现。原54项最终验收不减项，不将这些问题笼统记为精度不足。
- [华南虎/复旦源码审计](upstream_review.md)：区分实际启用机制与历史/注释功能；没有把对方成果或旧V5报告算作V6成功信用。

已验证的轮响应保持；腿部全构型响应仍是未qualified的工程先验。气簧/重力/闭链与现有通信、负载、COM等随机化贯穿实际物理链。RMCS未修改，工程探针或正式仿真启动均不构成硬件发布。TensorBoard读取Kaiser的`tensorboard-v6-active/v6-recovery-3051c6d`，本机入口为http://127.0.0.1:6006；正式PPO event文件已生成并写入指标。
