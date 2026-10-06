# source021 恢复探索：最小学习补救候选

只读冻结源 `d0dce6181df83b03fe166abcbadef3c81de2e57e`、本机安装RSL-RL 5.5.1和既有证据。本次没有SSH／GPU／PPO，没有改源码、合同、std、Adam或门。候选尚未实现／部署。

**先核现有6轴std／KL和完整新增100窗；若同封存配对支持探索阻碍接稳，再只消融entropy。** 不硬夹噪声、不用均值动作代替正式PPO采样、不动1s门。

## 已证实的训练语义

- `std_type='scalar'` 是6个独立、共享所有观察和课程的可训练std，直接clamp至[1e-6,1e6]后构造Normal；不是log-std或状态相关std。完整resume严格恢复保存值，构造时`init_std=1`不重设模型。
- REC3076六std为 `[.253705,.257971,.267365,.241563,.823053,.853682]`，POLICY顺序左髋、左辅助膝轴、右髋、右辅助膝轴、左轮、右轮。腿目标未限幅σ为0.0604–0.0668rad；轮速目标σ为8.23／8.54rad/s（轮缘目标0.494／0.512m/s）。这些不是实际速度σ，不能与实际0.25m/s门直接换算通过概率。
- 50Hz采样、200Hz反馈PD；raw动作由控制器确定性限幅／映射，原生PPO保存raw action、old log-prob及old(μ,σ)，更新重新求同一raw action的概率。此链保持不变。
- 冻结代码有效路线为5 epochs／4 minibatches、ratio clip0.2、γ0.99／λ0.95、entropy β=.005、mixed value coefficient2、actor grad norm1、fixed LR1e-4。安装版只有adaptive分支才用desired_KL=.01调LR；fixed没有KL自动刹车，ratio clip也不是硬KL上限。

\[
L=L_{PPO}+c_VL_V-\beta\bar H,\quad
H=\frac12\sum_i\log(2\pi e\sigma_i^2),\quad
\partial L_{entropy}/\partial\sigma_i=-\beta/\sigma_i.
\]

熵项给std增大的方向。REC3076腿轴熵loss导数约−.0187…−.0207、轮轴−.00586…−.00607；不是Adam位移预测，不能与每秒物理成本比较大小。

源码：[Normal／熵](/home/yukikaze/isaacsim60-venv/lib/python3.12/site-packages/rsl_rl/modules/distribution.py:165)、[PPO loss](/home/yukikaze/isaacsim60-venv/lib/python3.12/site-packages/rsl_rl/algorithms/ppo.py:285)、[配置覆盖](/tmp/scut-recovery-frozen-021-qv1ukmbr/scripts/train_chassis.py:529)。参数是冻结唯一代码路线；本次未SSH读远端agent副本。

## 成功少，不等于没有降std信号

在旧策略附近、ratio未clip的简化情况下：

\[
\partial L_{PPO}/\partial\sigma_i\simeq
-\mathbb E\left[A\left((a_i-\mu_i)^2/\sigma_i^3-1/\sigma_i\right)\right].
\]

若更大的动作偏差稳定对应更差优势，这项能降低std，不需要先有成功事件。当前HoldCost每20ms付款；精核饱和时，广域pseudo-Huber成本仍在。但PPO不反传物理导数，而靠动作—优势相关性。负reward不一定是负优势，支付−2.21/s不等于已给std下降梯度。

hold仍由actor含探索拥有；脚本、release和BLEND排除，actor导致的终止保留。mask只筛选样本／分段GAE，native概率、损失和Adam不变。全有效人口归一化优势；同一6个σ接受常规组、各层接稳和后续任务的共同梯度。当前24帧rollout=0.48s，1s稳定门跨rollout但物理稳定计时不重置；长期信用依赖critic bootstrap，即时成本仍有效。没有证实mask／GAE bug。

现有静态组`advantage_*`可能含排除后零样本且混lesson，不能当动态有效hold梯度。[分段与样本压缩](/tmp/scut-recovery-frozen-021-qv1ukmbr/src/wheeled_algo/recovery_rollout.py:78)、[所有权](/tmp/scut-recovery-frozen-021-qv1ukmbr/src/wheeled_tasks/chassis/recovery_training.py:212)。

## 下一步只读证据

1. 正式PPO继续，读取完整100actor CRC窗口现有 `DiagnosticUpdate/action_std_0..5`、KL探针／各组KL、RecoveryRollout有效人口、HoldCost、8个门、stable_peak与三lesson结局。6轴tag已经存在，无需新日志或重启；50／100重叠窗不当独立证据相加。
2. root的REC3076均值臂已完整600策略／2400物理步、正常exit0：L0释放443／成功216／1883初生，L1 157／63／675，L2 12／2／89；右删失保留。stochastic尚待真实结果与配对强验。均值臂部分成功不是行为或实车资格。
3. 若相同初生／RNG／std的stochastic明显更差，支持该封存模型的应用探索效应；不能认定仅轮轴、不能外推到最新PPO窗口。若mean也接不稳，则先核构型、支撑／真实接触、release／delay／跟随后目标，不用熵猜测遮盖物理错误。

如仍无法判别，下一冻结短诊断可只读原生有效hold／非hold、动态lesson的surrogate／entropy六轴梯度及clip fraction，复用native storage／loss；不第二次optimizer.step、不重采样、不改优势，保存恢复RNG与分布运行状态。该扩展尚未实现，不能热装正式进程。

## 单变量候选与双种子判决

仅在配对支持且完整100窗未见改善后，A保持β=.005，B取β=0作诊断端点。保留完整std／Adam／actor／critic；固定reward、LR、GAE、mask、物理、延迟、原1s稳定／真实跟随门和所有预算。β=0只是移走增噪激励，不保证std下降；不先指定目标σ。

当前入口没有读取合同`entropy_coef`，仅加JSON字段不会生效。未来候选需最小显式接线到 `cfg.algorithm.entropy_coef`、规范合同、runtime断言和CPU概率一致性测试，复用原生PPO。critic目标没变，若不新加预热，必须在迁移里明确remaining0／resume warmup0；不能默用v19合同迁移的10。现021的4新合同和10critic属于reward／goal迁移，entropy仍=.005，绝不能把其变化归因于熵消融。

A／B各2条独立PPO种子，预先冻结首个100actor点，共400actor全部计消费、不退弃选成本，必须有剩余预算且不超20k本域／121000总上限。序列执行，不能增设并行Kaiser owner。每seed内同一**冻结时最新own-REC**完整父和同一诊断RNG；不把正式线回滚到3076。两个seed的CPU／环境／CUDA随机流须真正不同：改seed标签后又恢复相同保存RNG不算独立。若需branch-only RNG fork，明确版本／派生SHA及批准边界，先核完整父恢复；原正式resume RNG不动。未批准／实现fork只报单种子。

CPU须验证实际β、完整初态、matched buffer的raw action／old概率／old分布／returns／advantages／mask不变、只改entropy loss、原optimizer.step只一次；检查解析std score／entropy导数、默认兼容和账本。之后同长度独立验证读首出生释放／成功／失败／超时／右删失、200Hz联合1s门、真接触、L0前向和正反yaw实际曝光／原1s跟随、完整基础静止／运动／高度／驻停及闭链峰值／3mm护栏。

**两条独立训练种子都改善stochastic接稳和原目标跟随，且基础／物理无实质退化，才把B列为下一合同候选。** 预先声明比较规则，逐seed报告分母和不确定性；只看std下降／mean成功／总reward、挑seed均不成立。结论仍不自动授质量资格或部署。

概率红线：不能发送 `a'=μ+c(a−μ)` 却按旧未缩放Normal记log-prob，也不能mean收集伪装stochastic。未来若做条件σ，collect、old log-prob／params、update、entropy、KL及采样时条件必须全一致；若用35D外的FSM条件还需承认新的策略／部署接口。本候选只改entropy，避开这整类扩展。

完整公式、SHA、缺证据与机器可读边界见同目录 `learning_remedy_design_021.json`。本设计不保证收敛时长、std下降或提速收益。
