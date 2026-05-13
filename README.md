# jerry_repository

HuggingFace Deep RL Course 学习笔记与代码。

## 目录结构

```
notebooks/
├── unit2/        # Q-Learning
├── unit4/        # Policy Gradient (Reinforce)
├── unit5/        # ML-Agents
├── unit8/        # PPO + Sample Factory (Doom)
└── bonus-unit3/  # Decision Transformer (本地 GPU 脚本)
```

## 各 Unit 简介

### Unit 2 - Q-Learning
从零实现 Q-Learning，在 FrozenLake-v1 和 Taxi-v3 环境中训练 agent。

### Unit 4 - Policy Gradient (Reinforce)
用 PyTorch 从零实现 Reinforce 算法（Monte Carlo Policy Gradient），基于 policy-based 方法直接优化策略。

### Unit 5 - ML-Agents
使用 Unity ML-Agents 工具包训练 agent，在 3D 环境中完成任务。

### Unit 8 Part 2 - PPO + Sample Factory
使用 Sample Factory（异步 PPO 实现）训练 agent 在 Doom 像素输入环境中完成目标收集任务。

### Bonus Unit 3 - Decision Transformer
将官方 Colab notebook 更新为可在本地 GPU 运行的 Python 脚本：
- 环境：HalfCheetah-v5（gymnasium 1.x + mujoco 3.x）
- 数据集：Minari expert-v0（替换过时的 D4RL v2）
- 支持 CLI 参数、GUI 实时渲染、视频录制
- 训练结果：6000 epochs，Episode Return 12,894（专家水平约 78%）

详见 [notebooks/bonus-unit3/README.md](notebooks/bonus-unit3/README.md)。

## 参考

- [HuggingFace Deep RL Course](https://huggingface.co/learn/deep-rl-course)
