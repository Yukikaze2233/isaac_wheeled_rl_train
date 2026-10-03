# V8 recovery takeover deployment evidence

2026-10-03, Kaiser. Release source: `5de8fb814c4b157c52858a4f38b8bad5788c4860`; plan SHA: `c77926685265ebc7700820abdb2230bfb8fb4d9ec8c6921c6e77187e29d97abb`.

正式续训已经执行。旧 recovery actor390／本域消费400／总消费14420完整封存；新 actor、critic、Adam 逐值恢复。10 次 critic-only 更新计入消费后，首 PPO 消费14431／actor391有九个actor张量改变。11:00（UTC+8）存活快照为总消费14454、本域消费434／actor414，其中本轮24次actor更新；不退款、不回滚旧A。

| 文件 | 证据范围 |
|---|---|
| `source_audit.json` | 旧V6与最新V5自起来源、物理时钟与资格边界 |
| `source_revision_009c.json` | 实际发布源码归档和v8计划认证 |
| `release_predeploy_009c.json` | 发布源码与30入口预检通过 |
| `migration_stop_009.json` | 原v7恢复worker合法封存 |
| `launch_009.json` | r9原始启动快照；启动当时尚未核实学习，后续证据在下一行 |
| `resume_verification_009.json`、`first_ppo_verification_009.json` | 完整状态精确恢复、首critic保持actor、首actor实际变化、不可变快照SHA、运行配置 |
| `live_native_audit_009.json` | 独立只读核验：唯一该root进程、16438个scalar tag有限、接管cost实际支付／普通flat为0、实际初态人口 |
| `live_progress_009.json` | 11:00实时PPO及远端TensorBoard6006存活快照 |
| `native_probe_009b.summary.json` | 初版12s工程探针发现楼梯crouched预备抬升越过原留台边界的风险；不是发布版结果 |
| `native_probe_009c.summary.json` | 最终发布版4096×240策略步／960物理反馈PD步，台阶池upright、真实hold／平台与actor终止责任核验 |
| `native_probe_009_failed.json` | 首探针未加载sim60 runtime导致场景初始化失败；无PPO消费，后续正确runtime启动修复 |

发布前运行候选009／009b分别来自`ebd53c2`／`65d4996`，未执行PPO；实际仅009c的`5de8fb8`进入正式续训。前两份源码认证回执保留原身份，不能用候选SHA替代当前运行源码。最终工程摘要保留完整远端报告路径与SHA，省略大型startup人口数组；原始报告仍在远端`recovery_takeover_probe_009b/report.json`和`recovery_takeover_probe_009c/report.json`，不修改历史记录。

CPU相关回归、30入口合同预检及原生工程核验通过。首PPO证明新版正在学习，工程探针不消耗PPO也不授予动态资格；短窗简单初态的2322/2509已结束episode比例不代表全域自起成功率。当前新合同学习质量尚未建立趋势；最新V5自起试验不是V6同条件验证，腿侧bag仍未qualified。

恢复任务65%＋完整flat41组35%常驻复习，初态reset-only按冻结origin400渐进；脚本／BLEND排除PPO，actor失败末步保留责任。其余五worker、父树、4096／35D81D／50Hz策略200Hz仿真PD、轮响应、每域20k与总121000保持。recovery actor到15000自动接后续域，无45项评测拦停。仅Kaiser运行TensorBoard，本机6006为SSH转发；6007属于其他实验。该结果文档提交不触发第二次部署。
