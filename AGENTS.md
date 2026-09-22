# 项目协作约定

## 训练指标与图表

- 训练曲线和图表数据统一使用 **TensorBoard** 记录与查看。
- 新增reward、loss、性能或行为指标时，接入现有RSL-RL logger的event文件，使用稳定tag和明确的step/单位。
- 默认通过TensorBoard比较实验；需要静态图片时，从event数据导出，避免建立独立的曲线数据源。
- checkpoint、合同、评测判定、归档manifest和SHA-256回执继续承担机器可读的恢复与验收职责。
- TensorBoard和SSH端口转发是独立监控进程，不能成为训练进程的生命周期依赖。

## 当前主线

- 业务模块按领域职责使用通用命名，例如`rewards.py`；参考项目来源写在注释和文档中，不作为新模块名前缀。

- V5入口和边界见`docs/V5_ARCHITECTURE.md`。
- 运行、TensorBoard访问、断连与回收说明见`docs/V5_INTEGRATED_TRAINING_20260921.md`。
- 远端运行使用冻结的源码归档；本地改动与提交不等于已部署到运行中的任务。
- 每轮训练迭代必须先提交已验证的训练代码、合同和计划，再从该commit冻结部署；启动回执记录commit与归档SHA。运行结果另行提交文档回执，不以未提交overlay代替训练版本提交。
