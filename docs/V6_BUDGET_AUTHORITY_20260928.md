# V6预算版本审计与新资产预算权威

日期：2026-09-28。范围：只读历史Git对象及本地冻结归档，冻结新资产预算设计；未启动训练、未修改历史合同或远端文件。

## 1. 500/800的准确来源

按各版本原始解析器与物化器复算全部阶段，参考环境数均4096、实际16384、rollout24。旧公式为`ceil(reference_updates * 4096 / 16384)`。

|版本|200Hz/precision入口P0参考更新|P0实际上限|全阶段实际上限|
|---|---:|---:|---:|
|736b745|2000|500|7000|
|e055a5a、12ba177|3200|800|7300|
|895ee92及审查时HEAD 1c0fd31|2000|500|7000|
|895ee92新增独立budget_retry入口|3200|800|7300|

`e055a5a`修改`v6_manual_remediation_v1.json`的`phase_schedule[0].updates`，通过继承污染旧入口。`12ba177`本身只修改assessment/progress容错，不是预算修改来源。`895ee92`已在`v6_scut35_continuation_v1.json`显式覆盖原2000，并将3200扩容放入独立`v6_scut35_budget_retry_v1.json`。

复算的precision阶段合同SHA前缀：旧500=`d6bc4c64`，12ba177下被改成800=`26b31275`，独立retry800=`34684663`；当前旧入口回到原`d6bc4c64`。两份本地源码tar分别匹配736b745和895ee92启动回执的归档SHA，实际读取的全部源码/合同与对应commit逐字节相同。

完整commit、传递依赖路径/原文件SHA、resolved计划及七阶段预算/合同SHA见[机器可读审计](../reports/v6_asset_review_20260928/budget_lineage_audit.json)。历史恢复依据仍是对应运行冻结合同和消费账本，不能用新HEAD追溯改写。

## 2. 123021的证据边界

“123021”在本机不是可解析的Git对象，也未找到可认证的完整运行标识。本轮Kaiser SSH返回`No route to host`，所以其具体远端归档身份仍标为`unverified`。没有把另一份500预算归档冒充123021；用户报告的500与已复现的历史机制一致，但不等于该具体归档SHA已经核验。

待连接恢复，应读取该run的完整路径、launch/source archive SHA、入口及resolved/materialized合同，按文件哈希核对。此项是历史产物补证，不决定新资产预算选择。

## 3. 新资产S1–S6独立冻结

唯一预算设计权威为[contracts/v6_new_asset_budget_v1.json](../contracts/v6_new_asset_budget_v1.json)，没有`extends_plan`，不继承旧`phase_schedule`。它明确标记为**预算规范，非可启动训练合同**。

- S1–S6采用20000环境、rollout24，参考环境数也为20000。阶段1000/2000/1000/1500/2000/1000均指实际优化迭代上限，含critic-only；不是旧4096参考口径。
- 正式总上限8500次、4,080,000,000 transitions；E0独立4096环境、四臂各200次，共78,643,200；合计4,158,643,200。
- 正式minibatch48000、10份、5epochs；E0为49152、2份、5epochs。回滚不返还消费，新增探针或更改环境数须显式另版。
- 可执行新合同须独立生成并显式映射S1–S6。物化后逐项比对该规范；若S1被解析成500、800或其他值，拒绝启动，不能选择“更像历史”的值继续。
- 完整合同与20000入口适配仍是新训练启动条件；G0离线资产实施可以先开展。不能把本次预算算术核验称为20000环境物化/训练已通过。

冻结运行版本时共同保存：预算文件SHA、解析器commit、所有实际读取的合同/引用文件SHA、resolved plan、各物化阶段合同SHA、CLI覆盖及已消费预算。顶层`contract_id`或JSON文件SHA不足以固定继承链。

## 4. 本轮验证

- 五个Git快照和两份本地归档按各自原始代码复算，没有用当前解析器代替历史代码。
- 使用`/home/yukikaze/isaacsim60-venv/bin/python`运行现有`tests/test_budget_resume.py`，4项通过，包括500/800预算迁移和语义变更拒绝。
- 新预算文件独立核对阶段合计、transitions、E0四臂和minibatch整除；验证回执见[budget_design_verification.json](../reports/v6_asset_review_20260928/budget_design_verification.json)。
- 未补做新资产仿真、GPU容量试验或实机实验。
