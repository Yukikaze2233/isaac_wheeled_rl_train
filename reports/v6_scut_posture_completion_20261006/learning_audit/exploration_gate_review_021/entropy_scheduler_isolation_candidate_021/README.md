# 研究调度隔离最小CPU候选

本目录只准备两个阻塞的隔离补丁，未应用、未部署、没有有效entropy试验。baseline为正式source021；只有 `scripts/run_task_campaign.py` 差异，生产文件与合同逐字不变。

`trial_state`由原共享claim恢复并深拷贝到内存，旧`campaign_state.json`不能覆盖或创建它。persist先把真实trial_state和原attempt账本写入claim，再写原consumption与本地report。无trial_state时普通账本/报告保持原字段与字节。这里保留的是未来严格准入产生的控制payload；仅检查payload为object，未提供试验准入、配额或容量作者入口。

研究发布需要显式transient `active_trial={trial_id,arm_id,run_directory}`，trial_id与已恢复claim一致且目录属于当前root的research。失去context、改目录、跨root或软链接逃逸会在收费/评审前拒绝。恢复旧report不会自动启动研究child；将来唯一原生serial queue负责在严格准入后设置/清除此context。

研究publish仍调用原account/update_attempt/persist计费，并在root progress明确active_trial及研究orchestration；四份startup/behavior/torque/live原文件留在其研究目录，不覆盖主线版本。它不执行原main assess/review dispatch，直接review_request/assess_request也有同一防线。main `_active_review`、pending_review、request/token历史及三个评审文件保持；清除context后原主线review可继续消费原合法请求一次。

原flock、单child execute、update_attempt单调收费、run/continuous/selector、seal、write_json/fsync及六域预算均未修改。没有第二调度器。

24项CPU通过，含原实现确实丢claim控制状态、原实现确实向研究child误发main SIGTERM两条负对照，以及claim提交后本地consumption失败、stale report伪状态、单调attempt消费、main token/文件隔离、直接review/assessment、陌生目录/ID、软链接逃逸和普通publish/persist逐字兼容。SIGTERM复现使用可控的合法验证stub，不声称做过原生评审/训练。补丁已git apply --check，仅检查未应用。

尚未实施：严格不可变manifest/admission与recipe SHA认证，按真实attempt派生arm剩余容量，注册研究progress重放与export清单，四arm串行队列，完整strict AB合同继承，postrestore真正RNGfork/private state resume。这份补丁单独不允许启动440试验，也不授权热改source021。原设计revision2与协议字节保持不变；root仍优先SIDE机械单变量。

`qualification=null`，0PPO，无SSH/GPU/App。
