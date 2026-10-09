# source031 实际暂存与 CPU 预检：独立核对

通过。独立重新流式读取完整 Git 归档，对全部 **4,343** 文件的字节数、SHA、Git/归档权限逐项核对本地 manifest 与实际远端 stage 回执，两者完全一致；归档 commit 为 `ef2cf6cd9aceeb74b6a5fc6e7dca7412a7a23f4a`，SHA `0b942599d452dc17b9a710349262fcd90218a9c427d3e07df85a55ecdfde59f6`。276 份 source029 旧合同逐项与真实 e5bd9 Git blob 的 SHA 一致，其中 36 份 v24–26 专项保持，12 份新 v27 对象与生成回执一致。

jump/stepjump 的 recipe flag、train contract 与 runtime 零目标标记一致；speed/terrain 无该开关，speed recipe 精确继承 v26。新落地模块及独立预期模块属于这次实际暂存的冻结文件集。

实际远端 CLI 指向新 source031 的 v27 plan，`--preflight-only --device cpu --num-envs 4096`；原入口在创建 `TaskCampaign` 及账户前直接执行静态校验并返回。真实执行 exit 0，stdout 为 `{"status": "passed", "roles": 30}`，原 log SHA `f6bafbd166fab9f444d3574be71f3d382753626ae8db5be49d94904feb2f40b9`；44.406 s 为运输调用与预检的外层墙钟，不能当作 PPO 更新耗时。

实际 stage SHA `b7dae15f3878aa92039580387bc98e526078fc4f617064a35c18da8994f93828`，preflight SHA `9c14b08e90dc4eb7a8f3933ba29035e025b3b36083cae37de64002445bc9a83f`，helper SHA `663efe61f1b75ced98cffe5077a5568069b9ee9b7dd03a333ff1dd5dfd4b94e9`，均与读回绑定一致。

初次 transfer001 因 NEW 错写旧 source030 路径而在 reservation 前拒绝，exit 1 与 stderr 留存。后续只读读回确认 5 个新 source031 路径均不存在、原三训练 PID/starttime 仍在，随后 distinct transfer002 才完成。读回中 `original_source030_preserved` 只检查旧目录存在，不是对旧树重新做完整字节认证；本报告不扩大该字段的证明范围。

2026-10-09 00:46:44 UTC 的只读读回确认新 stage 未绑定动态父，source031 boundary intent 与 revision receipt 均不存在。它未另行读取共享 entry 或最新账户，等待进程安全取消及未入场资格仍须其专用 guard。当前只证明新源暂存及静态预检成功，不能称已开始新 PPO 或学会跳阶。

机器回执 `source031_static_stage_independent_review001.json` 保存逐项结果及输入 SHA。B 本次没有 SSH、信号、新场景、PPO、远端改写或重复 CPU 测试；`qualification=null`。
