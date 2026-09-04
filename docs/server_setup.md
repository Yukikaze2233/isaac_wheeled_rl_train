# 服务器训练环境搭建

训练侧完整环境:Ubuntu + NVIDIA 驱动 + Isaac Sim + Isaac Lab + 本仓库。
照抄即可跑通;文末附云端租用与日常工作流。

## 1. 硬件与系统

| 项 | 最低 | 推荐 |
|---|---|---|
| GPU | 8 GB 显存(CUDA capable) | 16 GB(4090 / 5070 Ti 级) |
| 系统 | Ubuntu 22.04(配 Isaac Sim 4.5 + Isaac Lab 2.1) | Ubuntu 24.04(Isaac Sim 5.1 + Isaac Lab 2.3) |
| 磁盘 | 50 GB 空闲(Isaac Sim 本体 + 数据集 + checkpoint) | 100 GB |
| CPU/RAM | 8 核 / 32 GB | 16 核 / 64 GB |

云端参考:AutoDL 等 4090 实例(约 2 元/小时);flat+rough 全流程预算 300–500 卡时。

## 2. 版本矩阵(严格按此配,混版本是第一坑)

| 组件 | Ubuntu 22.04 路线 | Ubuntu 24.04 路线 |
|---|---|---|
| Isaac Sim | 4.5 | 5.1 |
| Isaac Lab | 2.1.0 | 2.3.x |
| Python | 3.10/3.11 | 3.11 |
| PyTorch | 2.7.0+cu128 | 2.7.0+cu128 |
| rsl-rl-lib | **==2.3.3(钉死)** | **==2.3.3(钉死)** |

> rsl-rl 警告:PyPI 最新版已是 3.x 新 API(TensorDict / actor-critic 键名),
> 与 Isaac Lab 2.x 的 wrapper 不兼容。验证方法:
> `python -c "from rsl_rl.runners import OnPolicyRunner; import inspect; print(inspect.signature(OnPolicyRunner.__init__))"`
> 应显示 `(env, train_cfg: dict, log_dir, device)`。

## 3. 安装步骤

```bash
# 1) NVIDIA 驱动 + CUDA 12.8(驱动 ≥ 550)
nvidia-smi   # 确认驱动与 GPU 可见

# 2) Isaac Sim(pip 方式,推荐服务器)
python3.11 -m venv ~/isaaclab_venv && source ~/isaaclab_venv/bin/activate
pip install isaacsim[all,extscache]==4.5.0.0   # 24.04 路线换 5.1
# 或官方 Workstation 安装(带 GUI 的机器):参见 Isaac Sim 官方文档

# 3) Isaac Lab(与 Isaac Sim 版本配对)
git clone https://github.com/isaac-sim/IsaacLab.git && cd IsaacLab
git checkout v2.1.0        # 24.04 路线: v2.3.x
./isaaclab.sh --install

# 4) 本仓库(三包一次装齐,src 布局)
git clone https://github.com/Yukikaze2233/wheeled-biped-rl-train.git
cd wheeled-biped-rl-train
pip install -e .           # 装 wheeled_world / wheeled_tasks / wheeled_algo
pip install rsl-rl-lib==2.3.3

# 5) 资产
export WHEELED_RL_ASSETS_DIR=/path/to/your/usd_dir   # 见 assets/README.md
```

## 4. 验证(按顺序,每步过了再下一步)

```bash
# a) 栈完整性
python -c "import torch, isaaclab, rsl_rl; print(torch.__version__, torch.cuda.is_available())"
python -c "from rsl_rl.runners import OnPolicyRunner"          # 2.3.x 签名检查(见上)

# b) 纯 torch 套件(不启仿真,~8 min)
PYTHONPATH=src:tests python tests/test_mdp.py
PYTHONPATH=src:tests python tests/test_algorithms.py
PYTHONPATH=src:tests python tests/test_rsl_rl_branches.py

# c) 仿真启动 + 资产检查(无训练)
./isaaclab.sh -p scripts/train.py --task WheeledBiped-Flat-v0 \
    --num_envs 64 --max_iterations 5 --headless     # 5 迭代冒烟

# d) 全规模
./isaaclab.sh -p scripts/train.py --task WheeledBiped-Flat-v0 \
    --num_envs 4096 --max_iterations 20000 --headless
```

## 5. 日常工作流(云端)

```bash
# 训练放后台(nohup / tmux)
nohup ./isaaclab.sh -p scripts/train.py --task WheeledBiped-Flat-v0 \
    --num_envs 4096 --max_iterations 20000 --headless \
    --log_dir runs/flat_v1 > train.log 2>&1 &

# checkpoint 续训(调参闭环 ~1h 级)
... --checkpoint runs/flat_v1/model_latest.pt

# 导出 ONNX 拉回本地做 sim2sim
./isaaclab.sh -p scripts/export_onnx.py \
    --checkpoint runs/flat_v1/model_final.pt --output policy.onnx
# 本地: python3 tools/check_onnx_contract.py policy.onnx (部署仓)
```

## 6. 常见问题

| 症状 | 原因/处理 |
|---|---|
| `rsl_rl` API 报 TensorDict / actor-critic 键名错误 | 装了 3.x,`pip install rsl-rl-lib==2.3.3` 降级 |
| Isaac Sim 启动即崩 | 驱动版本不匹配;`nvidia-smi` 确认 ≥550,或换 Isaac Sim 版本 |
| 显存 OOM | 降 num_envs(4096→2048→1024);关 render(headless) |
| 闭链脱链(并联腿) | 提高物理频率比加 solver 迭代有效(经验:200→500 Hz) |
| 训练发散 | 先窄 DR 训通再放宽;从 checkpoint 续训;检查 reward 量纲 |
