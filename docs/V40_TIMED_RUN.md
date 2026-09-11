# V40 定时预算、可靠收尾、显式回传（限定已批准研究模型）

## 当前优先级与不可绕过的门

用户已明确批准 **串联等效研究模型及通过独立 tmux 部署/训练**；默认契约选择独立的
`research_manifest.json`，许可仅限已记录的六对关节内部接触排除。原 `manifest.json`
材料审查仍 false，CAD/部件归属未修复，不代表硬件部署许可。`--research`、审批记录、
原文件哈希、精确过滤范围、静态碰撞门和运行时版本/源码门均不可跳过。
本次启动修复不改资产、契约、env/rewards/PPO 或导出实现。旧 NoNaN / 有限数检查不算通过：
预算耗尽、一次 update 完成、ONNX 数值对齐，都不等于策略收敛或真实物理验证。

用户已设置平台 8 小时关机。代码 **不设置、更改或取消 AutoDL 关机**，不执行 shutdown，
也没有云文件通道。**实际平台关机日期、时刻、时区尚未给出，必须向用户确认，不能猜。**

## 8 小时里至少留下 30 分钟

设用户确认的平台硬关机时刻为 S，训练截止 C 应满足 **C <= S - 30 分钟**；如果模型、
网络、初始化或导出较慢，要留更多。30 分钟是计划余量，不是完成保证。

- 理想完整 8h 窗口的训练预算上限示例是 27,000 秒（7.5h），**不是当前剩余预算**。
- 相对预算从 learn 前开始，不计 preflight、AppLauncher、场景初始化；所以仅设置 27,000
  不能保证赶上已有的平台定时关机，必须再给真实时区化的绝对 C。
- 多轮必须共用不晚于 C 的绝对截止，每轮重新估算剩余时间；不得每轮重置一个新的 8h。
- 应在 C 前启动本机等待回传，确保完成 receipt 出现后马上拉取。平台已先关机且 SSH
  不通时，只能明确报告未回传/等待开机，不能声称文件已到本机。
- tmux 仅避免终端断开导致退出，**抵抗不了平台关机、SIGKILL、断电或 GPU/C 调用卡死**。

## 训练 CLI/API 改动

仅新增可选参数，旧 CLI、真 resume/finetune、Python3.11、精确版本（含 CUDA local tag、
5.1.0 与 5.1.0.0 等价）、metadata hook、严格碰撞门都保留：

| 参数 | 语义 |
|---|---|
| --max-runtime-seconds N | 有限正数；本轮训练阶段相对软预算。默认无相对预算 |
| --stop-at ISO8601 | 必须带 Z 或 ±HH:MM 时区的完整日期时间；默认无绝对预算 |
| --max_iterations / --max-iterations N | 默认 20,000；**本次追加**的迭代数，resume 不解释为历史总目标 |
| --run-dir PATH | 必须是新目录；resume/finetune 也不得覆写旧 run |

### 训练长度参考与首测限制

按已核对的实际配置覆盖关系：

| 参考 | 并行环境 | 每轮每环境步数 | 策略频率 | 训练长度口径 |
|---|---:|---:|---:|---|
| SCUT | 4096 | 24 | 50 Hz | agent.yaml 上限 20,000；所参照 checkpoint 为 13,000 |
| 复旦 WheelLegged | 8192 | 48 | 100 Hz | plane 基类 5,000 被实际 WheelLeggedCfgPPO.runner 覆盖为 50,000 |

V40 train CLI 默认上限调整为 **20,000 本次追加 iterations**，不是照搬展示 checkpoint 的
13,000，也不是误用复旦基类的 5,000。不同并行环境数量、rollout 长度、频率与硬件吞吐
使 iteration 数不能直接当作训练时长；**20,000 不保证收敛，也不保证在 8 小时内跑完**。
绝对截止和训练预算仍优先，必须为保存、导出、回传留至少 30 分钟。

真正首测将显式给 --max-iterations 2 或 100，并配短 --max-runtime-seconds / 明确 --stop-at，
**不得把新的 20,000 默认值直接当成首次开跑命令**。研究等效模型的许可、独立 research
manifest 审核及接线由资产/主任务负责；本预算实现不改原 CAD，也不自行将任何 passed 改 true。

确认实际截止 C 后，由 tmux 启动的训练命令形状如下（占位符不能直接运行）：

~~~text
/ABS/ENV/bin/python /ABS/REPO/scripts/train_v40.py \
  --research --headless --stage stand --run-dir /ABS/NEW_RUN \
  --max-iterations ITERATIONS --max-runtime-seconds REMAINING_TRAIN_SECONDS \
  --stop-at ACTUAL_TRAIN_CUTOFF_WITH_TIMEZONE
~~~

resume 使用 --resume /ABS/OLD_RUN/model_final.pt，恢复模型/optimizer/stock iteration；
finetune 使用 --finetune 同样路径，只用权重。二者互斥，均要求匹配当前 contract/assets。
没有完成本次 update 的 resume 调用仍标记 **本次 untrained**，不会把输入 checkpoint
重新命名成一次成功训练的 model_final.pt。

### 安全边界

实现位于 src/wheeled_algo/v40_job.py：

1. 绝对截止在 preflight 后、AppLauncher import 前后以及 learn 前再次检查；过期不启动
   新仿真。若场景初始化期间过期，则 learn 前停止，0 更新收据为 untrained。
2. 从启动到 cleanup，SIGINT/SIGTERM handler **只置 flag**，不保存、不抛异常、不强杀 C
   调用。真正的 stop 仅由官方 RslRlVecEnvWrapper **同一个实例**的 step 调用前触发。
3. runner.alg.update 只包一层计数：原函数正常返回才 +1；不插入 optimizer 检查或改 PPO。
   信号/预算在 update 中到达会等待完整返回，再在下一次 step 前处理。
4. 因此相对/绝对软预算可以超出一个正在执行的 step/update、日志或同步操作；不是硬实时
   deadline。不通过“中断半个 optimizer update”来假装准时。
5. 计划停止时，即使已有部分 rollout，保存的仍是上一次完整 update 的模型/optimizer。
   既有规范要求 normalization 关闭，未改变 PPO 学习逻辑。
6. 任意学习异常，尤其 update 抛错，不保存半更新 final；保留已有周期 checkpoint、原异常
   traceback，并尝试写 training_failed 收据。磁盘/收据错误不掩盖原学习异常。
7. final 保存仍调用 runner.save，所以所有身份信息继续通过原 metadata hook；先写私有
   partial，再无覆写原子发布 model_final.pt。保存失败标记 save_failed。
8. env.close 与 simulation_app.close 为嵌套 finally，env 关闭出错也仍尝试 app 关闭。
   如初始化早期失败、断电或存储损坏，可能无 completion；缺收据不是成功。

### 导出和终态 receipt

正常迭代完成/预算停止/用户软停止且至少有 1 次完整 update 时，用同一绝对解释器启动
隔离 CPU 子进程，调用既有严格导出 CLI，而非在活跃 Kit 进程内直接调用导出 API：

~~~text
/ABS/ENV/bin/python /ABS/REPO/scripts/export_v40_onnx.py \
  --checkpoint /ABS/RUN/model_final.pt --run-manifest /ABS/RUN/run_manifest.json \
  --output /ABS/RUN/policy.onnx
~~~

子进程使用 AppLauncher import 前保存的环境/cwd，强制 `CUDA_VISIBLE_DEVICES=-1`，
不继承 Kit 后续修改的 Python/动态库路径。独立进程组、900 秒导出超时和 run 内独占的
`export.stdout.log` / `export.stderr.log` 保留失败证据；这两份日志不扩大回传白名单。
父进程要求退出码 0，并复核 sidecar、身份及产物哈希；即使已生成文件，退出崩溃仍算失败。
严格导出检查不变：weights-only checkpoint、完整 actor/critic、身份 manifest/hash、固定 ONNX
接口、checker 和 9 个 CPU ORT 对齐样本均不可跳过。导出失败保留 pt，标记 export_failed，
不把存在一个 onnx 文件当作导出成功；可能残留的非完成 ONNX 不在回传白名单。

最后无覆写原子发布 completion.json，至少含：

- complete=true：本轮终态记录已发布，**不表示成功训练**；
- completed_updates：本次正常返回的完整 updates，不依赖旧 stock iteration 编号；
- requested_iterations / requested_iterations_completed：追加目标及是否全部完成；
- stop_reason：iterations_completed / max_runtime_seconds / stop_at / sigint / sigterm /
  soft_stop / training_exception；
- status：completed / stopped / untrained / training_failed / save_failed / export_failed；
- export_status：verified / export_failed / not_attempted；
- update_failed、error、训练阶段 elapsed、预算/截止和完成时间；
- **policy_quality_verified=false**：永不由定时完成或数值对齐推断策略质量；
- artifacts：闭合文件白名单，每项准确的 path、size、SHA256。

正常全部迭代 + 导出成功是 completed；计划提前停止 + 完整模型 + 导出成功是 stopped，
不是收敛。上述两者 CLI 返回 0；untrained/save_failed/export_failed 返回 3；启动门拒绝
返回 2；训练异常保留异常退出。requested_iterations_completed 可在 export_failed 时为
true，明确区分“updates 全部做完”和“导出成功”。

### 白名单与快照边界

成功模型目录中可回传的八个固定扁平文件：

~~~text
model_final.pt
policy.onnx
policy.onnx.json
run_manifest.json
agent_config.json
contract.json
asset_manifest.json
source_hashes.json
~~~

contract.json 保存原始 JSON 字节，不重排键/空白；保存前校验其对象与已加载 contract
一致、语义 digest 一致。asset_manifest.json 同样保留原始字节与已验证 raw-file SHA。
source_hashes.json 含固定 V40 训练/控制/导出/启动/回传/资产生成源码的 SHA256、资产
manifest 的 files_sha256 及 contract 原始快照 SHA。

**不是完整资产快照**：不递归打包 CAD/STL、仓库、环境或任意用户目录；必须单独留存匹配
hash 的已审核仓库/资产。文件列表不由远端随意扩展，不带 .env / .ssh / .git、用户任意
文件、周期 checkpoint 或 TensorBoard/code diff 日志。ONNX 部署还需要匹配资产及合同，
本白名单本身不承诺能从零重建训练环境。

export_failed 时仅回传五个来源 JSON + 完成 pt；失败/untrained 不自动回传“模型”。
completion.json 作为收据本身单独保存并记录 hash，避免自引用 hash。

## 本机显式回传

scripts/pull_v40_artifacts.py 仅使用本机已安装 DSH 受管 SSH HTTP API；路由已只读核对
本机 @linxin666/dsh-ssh/lib/index.js，不使用裸 SSH、密码、私钥或另开 browser：

- POST /api/dsh-ssh/exec，JSON {alias, command, timeoutMs}，响应 {result: ...}；
- GET /api/dsh-ssh/download?alias=...&remotePath=...，返回二进制流。

默认管理地址 http://127.0.0.1:3080；--management-url 只能是 loopback HTTP(S) origin，
禁凭据/path/query/fragment、代理和重定向；localhost 规范成 127.0.0.1。

~~~text
LOCAL_PYTHON scripts/pull_v40_artifacts.py \
  --alias EXPLICIT_CONFIGURED_ALIAS --remote-run-dir /ABS/REMOTE_RUN \
  --remote-python /root/autodl-tmp/scut-isaac51-lab230/py311/bin/python \
  --destination /ABS/NEW_LOCAL_DESTINATION \
  --wait --poll-interval 30 --max-wait-seconds EXPLICIT_WAIT_BUDGET
~~~

--alias、--remote-run-dir、--destination 必填。不带 --wait 时只检查一次；完成则拉取，
未完成明确退出。--wait 只重试未出现 receipt/网络不可用；轮询间隔必须 >=30s，默认最大
等待 3600s。等待预算用于 **等 completion**，不是下载总时限；HTTP 超时另有限制。
坏 receipt/路径/哈希不会通过无限重试隐藏。

可选 `--remote-python` 仅选择只读 stdlib probe 的解释器，不启动训练；不传时保留 `python3`
默认值。远端非交互 shell 的 PATH 没有 python3 时，显式指定规范绝对 POSIX 路径（如上），
无需激活环境或依赖远端 PATH；含空格的路径在本机命令中加引号，远端参数会安全引用。
命令明确非零退出（如解释器不存在）直接报错，不提示等待开机，也不因 `--wait` 重试。

安全规则：

- 远端 run 必须是规范绝对 POSIX 路径；成员只能来自固定文件集。绝对成员、..、目录、
  反斜杠、“看起来像链接”的描述、未知文件/字段式路径都拒绝。
- 只读 probe 用逐级 O_NOFOLLOW 打开目录/文件；拒绝 symlink、hardlink、FIFO、设备。
- 下载前逐次复核，下载后再次复核 inode/ctime/size 指纹及 completion；本机逐文件
  写 .partial，长度与 SHA256 全通过后才无覆写发布。
- DSH SFTP /download 本身不支持 nofollow 原子事务。前后指纹 + SHA 能检测普通并发
  修改，**不是恶意受管服务器的安全隔离证明**；受管主机必须可信、完成产物应保持只读。
- 目标目录必须完全不存在，父目录必须已存在且每一级都不能是 symlink。包括之前失败
  留下的 partial 目录也不自动复用，更不覆盖未知已有数据。需要人工检查后选新目录。
- 失败不删除远端，可能留本机已验证文件和 partial，但没有 local_receipt.json 就不算
  完成回传。全部通过后才生成 local_receipt.json，明确 transfer_verified=true、
  run/export 状态和 policy_quality_verified=false；不等于训练或硬件验收。
- export_failed 的 pt-only 回传可以成功，local receipt 仍明确 export_failed，绝不伪称
  有已验证 ONNX。平台 SSH 不通时明确失败/等开机，不假设云文件可取。

## 独立 tmux 启动

用户服务器已配置 tmux。scripts/start_v40_tmux.py 默认 **dry-run、零网络**；只有显式
--launch 才通过同一受管 API 执行已批准范围。必须指定绝对 ENV/bin/python 和 repo。
预检子进程和 detached worker 的训练子进程均清掉旧 PYTHONHOME/PYTHONPATH，显式设置
目标 ENV/bin 与系统工具 PATH，并设 `ENABLE_CAMERAS=0`、`LIVESTREAM=0`、`PYTHONUNBUFFERED=1`。
`tmux new-session` 以多参数直接执行解释器 `-c worker`，不经 zsh/单字符串 shell-command；
会话级 `-e` 同时清空旧 Python 路径并固定上述环境，保护 worker 解释器自身启动。
不会安装、自动激活环境、修改 tmux 全局配置或重启服务端。

~~~text
LOCAL_PYTHON scripts/start_v40_tmux.py \
  --alias EXPLICIT_ALIAS --remote-repo /ABS/REPO --python /ABS/ENV/bin/python \
  --remote-run-dir /ABS/NEW_RUN --audit-dir /ABS/SEPARATE_NEW_AUDIT \
  --stop-at ACTUAL_TRAIN_CUTOFF_WITH_TIMEZONE --research \
  --max-iterations ITERATIONS --max-runtime-seconds REMAINING_TRAIN_SECONDS
~~~

以上不带 --launch，不会联机。确认目标、实际截止与预算后加 --launch：

1. 先用**同一绝对解释器与同一 train 参数**执行真实 --preflight-only，只有退出 0 且
   ready=true、simulation_started=false 才进入 new-session。train 本身仍再次执行门。
2. 生成 v40-UTC-UUID 独立唯一 session，不复用/干扰已有会话，不用 kill-server。
3. 不提前 mkdir run；只创建与 run 无祖先关系的独立 audit 目录（其父目录须已存在）。
4. audit 保存 preflight stdout/stderr、launch.json、train.log 和 exit.json；训练目录
   由 train 自己创建。exit.json 永不凭进程 exit 0 宣称训练成功。
5. 管理请求中断/超时后不盲目重试，以免开多个独立会话；人工检查 audit/session。
   会话消失、tmux new-session 返回 0 都不是训练完成。**唯一完成依据是 completion
   receipt + 白名单 hash**，回传完成另看本机 local receipt。

## 离线验证与未验证项

仅 CPU，使用现有解释器及已有依赖，不安装：

~~~text
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=src:../.rl_deps_rsl23 ../.venv_mj314/bin/python -B -m pytest -q -p no:cacheprovider \
  tests/v40/test_launch.py tests/v40/test_job.py tests/v40/test_env_contract_static.py \
  tests/v40/test_metrics.py tests/v40/test_export.py
~~~

test_job 覆盖 fake clock 边界/时区、flag-only 信号、完整 update 计数、0 更新无 final、
异常 update 不伪装正常结束、metadata 存盘、原始合同/资产 JSON 快照、严格导出实际 API
与合成权重的 ORT 对齐、坏 receipt/路径/symlink/hardlink/哈希/覆盖防护、动态 loopback
fake HTTP 小文件回传、等待预算，以及全部 subprocess mocked 的 tmux preflight gate。

测试没有真实 PPO 学习；合成权重导出不是训练结果，fake ONNX 小文件只是回传协议 fixture。
没有连接真实 3080/SSH 别名、实际模型下载、安装、远程 tmux 启动或关机操作。没有把真实
collision manifest 改 true。未验证 IsaacLab/IsaacSim/GPU 集成、真实信号到达 C/CUDA 的
延迟、真实训练收敛、平台倒计时、长时间保存导出耗时/网络吞吐、真实受管 SSH/tmux。
当前许可仅覆盖已批准研究模型；实际关机时刻/剩余预算仍须确认，服务器上传和集成验证
由主任务负责。本地启动修复不执行远程操作，也不把离线测试当作已开训证据。
