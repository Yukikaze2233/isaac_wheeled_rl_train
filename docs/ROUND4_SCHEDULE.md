# Kaiser 零点后按修复资产就绪状态启动

## 已设置的任务

- 时间：**北京时间2026-09-15 00:00起**；本机和Kaiser均已核对为UTC+08:00。
- 用户选择：等待修复后的动力学模型，不能以旧串联等效资产替代。
- 目标：新模型验证后，从零启动1024env、30000更新的正式任务。
- 当前事实：几何预览可运行，但修复后动力学尚未登记为就绪；**这不是保证零点已能启动训练**。
- Kaiser用户cron每分钟调用一次检查，cron服务已确认active。SSH断开不影响服务端检查；WSL需要处于运行状态。WSL停止时不会自动唤醒，重新启动且cron运行后继续检查。
- 没有预设等待截止日。未就绪继续等待；成功提交或提交失败后停止自动尝试，防止重复训练。

任务目录：

```text
/home/kaiser/robot-rl-sim60/scheduled-training/round4-repaired-20260915/
  schedule.py
  request.json
  schedule.lock
  state.json
  cron.log
```

实际request副本：[round4_repaired_schedule_20260915.json](evidence/round4_repaired_schedule_20260915.json)。

已观察到cron自动将`state.json.checked_at`更新到北京时间19:41:01；此前手动初始检查为19:38:35，状态均为`scheduled_waiting_time`。因此已确认周期调用实际发生，不是仅生成了命令文本。

## 状态与一次性启动

1. 零点前：`scheduled_waiting_time`，不执行部署/训练入口。
2. 零点后缺少`ready.json`：`waiting_repaired_asset`。
3. 登记文件、证据或部署校验未通过：`blocked_readiness`，保留具体原因，下次可重试检查。
4. 全部校验通过后，先写`launch_claimed`，再调用已绑定部署中的`scripts/round4/launch.py --launch`。
5. 提交成功：`submitted`；只代表训练worker已提交，不代表PPO已完成或效果通过。提交失败/中断需检查现场，不能自动重试制造第二个run。

锁防止两个cron tick同时提交。当前用户crontab的已有任务保留，只新增带`robot-rl-schedule:round4-repaired-20260915`标记的一行。结束或取消后可移除该标记行；审计文件保留。

## 修复资产完成后的交接协议

由完成修复与验证的部署步骤原子写入任务目录的`ready.json`，不要仅因几何窗口能动就填写通过。所需字段：

```text
git_commit              完整40位已提交commit
snapshot_sha256         该部署snapshot.json的SHA256
validation_path         修复动力学验证报告的绝对路径
validation_sha256       上述报告的SHA256
geometry_source_path    部署侧保存的用户确认chassis.usdc几何源
ground_usd              本机已缓存官方地面USD的绝对路径
ground_sha256           上述地面文件的SHA256
```

部署位置须为`<deployment_root>/experiments/round4-full-<git_commit>/`，代码叶名为`isaac_wheeled_rl_train`。scheduler逐项核对snapshot登记源码/资产哈希，再核对验证报告：

- `scope`等于`v40_repaired_dynamics`，`passed`为true。
- `checks`中`kinematic_closure`、`dynamic_constraints`、`mass_properties`、`actuator_mapping`、`height_domain`均为true，并有实际验证结果支撑。
- `contract_file_sha256`、`asset_manifest_sha256`与该部署实际文件一致。
- 明确拒绝旧资产manifest摘要`df5ca769…4b364886`。
- 必须绑定`model/纯底盘/chassis.usdc`的几何源SHA`4d502e88c5dcb32ad3f88e3c9bbcf2dc8aede3b615d27fbcdf69b5202d3099b9`，核对实际源文件及验证报告的`geometry_source_sha256`。这是动力学模型的几何来源，不把几何USD自身当作可训练资产。

几何包已同步至Kaiser的`/home/kaiser/robot-rl-sim60/model/纯底盘/`，模型、演示场景、运动学数据和附带工具均逐项size/SHA通过。该包仍标记`physics_ready=false`；调度器代码和request已更新为要求上述几何指纹，未创建训练就绪文件。

随后调用部署launcher的dry-run校验，必须得到同commit、scratch、parent=null、1024env、30000更新的计划，且命令不含旧权重初始化选项。只有通过后才正式提交。

**目前没有创建ready.json，也没有登记任何旧七刚体训练作为替代。** 修复后的合同、完整动力学验证和适配后的launcher仍须完成；这个协议只负责消费其结果，不自行推断CAD参数或生成虚假的验证记录。

## 查看与验证

查看服务端`state.json`和`cron.log`即可；无需重复安装。安装器可重入，同id不同命令会报错而不覆盖。

```bash
python3 /home/kaiser/robot-rl-sim60/scheduled-training/round4-repaired-20260915/schedule.py \
  tick --request /home/kaiser/robot-rl-sim60/scheduled-training/round4-repaired-20260915/request.json
```

CPU测试15项通过：零点前不启动、时区、未就绪等待、无隐式截止、旧资产/源码/合同/确认几何篡改与质量证据缺失拒绝、成功/失败后一次性语义、保留已有cron任务。测试使用合成回执，不是动力学资产验收。未运行新的PPO训练。
