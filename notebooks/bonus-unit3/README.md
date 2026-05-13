# Decision Transformer - HalfCheetah

HuggingFace Deep RL Course Bonus Unit 3 的本地运行版本。

原始 notebook 针对 Google Colab 编写，依赖已过时。本项目将其更新为可在本地 GPU 环境运行的 Python 脚本。

## 环境要求

- NVIDIA GPU（已在 RTX 4080 Laptop + CUDA 12.8 上验证）
- Miniconda / Anaconda
- Python 3.10

## 安装

```bash
conda create -n dt-learn python=3.10 -y
conda activate dt-learn

pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install "gymnasium[mujoco]>=1.0.0"
pip install transformers accelerate
pip install "minari[hdf5]"
pip install moviepy
pip install huggingface_hub
```

> 如果 CUDA 版本不是 12.8，替换安装命令中的 `cu128`，参考 [PyTorch 官网](https://pytorch.org/get-started/locally/)。

## 运行

```bash
# 默认训练（6000 epochs）+ 评估
python train_decision_transformer.py

# 自定义 epochs
python train_decision_transformer.py --epochs 6000

# 仅评估（从已有 checkpoint 加载，不重新训练）
python train_decision_transformer.py --eval_only

# 后台运行
nohup python train_decision_transformer.py --epochs 6000 > train.log 2>&1 &

# 查看训练进度
grep "loss" train.log | tail -5
```

训练完成后视频保存在 `./video/`。

## 主要参数

| 参数 | 值 | 说明 |
|---|---|---|
| `num_train_epochs` | 6000 | 对应约 96,000 梯度步 |
| `per_device_train_batch_size` | 64 | |
| `learning_rate` | 1e-4 | |
| `TARGET_RETURN` | 16.0 | 16000 / scale(1000) |
| `CONTEXT_LEN` | 20 | 自回归推理的上下文窗口 |

## 实验结果

| 训练步数 | epochs | train_loss | Episode Return | 专家水平 (~16500) |
|---|---|---|---|---|
| 1,920 | 120 | 0.097 | ~33 | 0.2% |
| 96,000 | 6000 | 0.055 | **12,894** | **78%** |

## 相对原 notebook 的主要更新

原 notebook 使用 `gym==0.21` + `mujoco_py` + Colab 专属渲染，均已过时。更新内容：

| 问题 | 原版 | 本版 |
|---|---|---|
| 训练数据 | D4RL `halfcheetah-expert-v2`（pkl，物理引擎不兼容）| Minari `mujoco/halfcheetah/expert-v0`（v5 原生数据）|
| 环境 | `HalfCheetah-v3`（gym） | `HalfCheetah-v5`（gymnasium 1.x）|
| 物理引擎 | `mujoco_py`（已废弃） | `mujoco` 3.x（官方 Python 绑定）|
| 渲染 | `colabgymrender`（Colab 专用）| `gymnasium.wrappers.RecordVideo` |
| 数据加载 | `datasets.load_dataset`（v4 不再支持脚本） | `minari.load_dataset` |
| `max_length` | `model.config.max_length` | `CONTEXT_LEN = 20`（transformers 5.x 移除该属性）|
| 设备 | 硬编码 `cpu` | 自动检测 `cuda` / `cpu` |

## 数据集

训练数据由 Minari 在首次运行时自动下载，缓存在：

```
~/.minari/datasets/mujoco/halfcheetah/expert-v0/   # 约 201MB
```

数据集信息：
- 来源：`mujoco/halfcheetah/expert-v0`（Farama Foundation）
- 内容：1000 条专家轨迹，每条 1000 步，共 100 万个 state-action 对
- 环境：HalfCheetah-v5，专家 return ≈ 16,500

> Minari 使用独立缓存目录 `~/.minari/`（而非 HuggingFace 缓存），可通过环境变量 `MINARI_DATASETS_PATH` 修改存储位置。

## 文件说明

```
bonus-unit3/
├── train_decision_transformer.py   # 主脚本（训练 + 评估）
├── 101_train_decision_transformers.ipynb  # 原始 notebook（仅供参考）
├── output/                         # 训练 checkpoint（最新一个，约 15MB）
└── video/                          # 评估录制的视频
```

## 参考

- [HuggingFace Deep RL Course - Bonus Unit 3](https://huggingface.co/learn/deep-rl-course/bonus-unit3)
- [Decision Transformer 论文](https://arxiv.org/abs/2106.01345)
- [Minari 数据集](https://minari.farama.org/)
