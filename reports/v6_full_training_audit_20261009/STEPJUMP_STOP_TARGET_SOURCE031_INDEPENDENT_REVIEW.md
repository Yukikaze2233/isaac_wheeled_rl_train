# source031 跳阶停车目标：独立代码与静态来源图审查

审查通过，无剩余源码接线阻断。修复代码为 `0f4b5f673e5f7e2cb382a8fcbd5766b0e13d8a9b`（父 81ad），实际生成的 v27 静态来源图提交为 `ef2cf6cd9aceeb74b6a5fc6e7dca7412a7a23f4a`，独立树干净。生成时执行源码准确为 0f4；ef2 是之后的合同提交。

重新计算并核对了作者 6 份冻结 Git blob、CPU stdout 尾部回执及付款 JSON 的 SHA。作者实际 focused CPU 为 17 passed / 31.94 s；最后来源登记修改后，同一真实物化正负例再过 1 passed / 23.97 s。B 没有重复运行测试。回执保留了前期 fixture 失败，未把它们误记为最终生产缺陷或通过。

数据流确认：真实 `StepJumpTask.enabled` 在新接受请求时决定落台 vx/yaw 零目标；原起跳命令仍控制接触前轮速，普通跳跃目标不变。3 s 非续期付款窗、6 s outcome deadline、0.35 s 平台稳定、0.25 m/s 平台速度门槛不变。recipe、train contract、独立 runtime expectation、实例 runtime marker 四处一致；仅未来 v27 jump/stepjump 开启，v27 两个实际变更模块已登记 source hashes。

付款 JSON 的独立公式重算与实际 float32 数值一致：锁定 +0.4 m/s、−0.2 rad/s 时，旧停车成本为 −1.1612496/s 与 −0.2410461/s；新跳阶 lane 停车为零，继续起跳目标仍扣相同成本；混合 batch 的普通跳跃 lane 精确保留。该证据只证明目标与成本一致，不证明 PhysX 触轮或策略学会。

本轮 root 授权的真实 `write_bundle` 生成了 12 个 v27 对象，11 个改变；speed expectations 的第 12 个对象逐字节不变。逐 Git blob 重算全部 276 份旧合同保持，其中 36 份 v24–26 专项对象保持。有效 flag/design/runtime/source inventory、speed 精确 v26 recipe、物理 ABI 和未来三域 phase audit 均在实际 canonical/static 配置上核对。

生成回执 SHA `b3d938cf145fff595a4999d25a906e41380866c1a834ae08161bd54e569f0eb1`；独立机器回执为 `stepjump_stop_target_source031_independent_review001.json`，SHA `1a97572c45abcdacda7fcc5426837fa512672fb4f8ba52bea627324f93478dcc`。具体原生采纳顺序见 `STEPJUMP_STOP_TARGET_ADOPTION_PATH_REVIEW.md`。

本次没有 SSH、远端信号、PPO、账户改写或活队列修改。仍待 root 认证旧 81ad 未入场与等待进程安全退休、独立新归档/stage/once 队列、真实正常 completed-speed 父证书、native runtime/collision 与固定任务回放。`qualification=null`，不授跳阶性能或实车资格。
