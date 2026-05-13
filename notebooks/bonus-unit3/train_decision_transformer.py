#!/usr/bin/env python3
"""
Decision Transformer - HalfCheetah training and evaluation.
Updated from HuggingFace deep-rl-course bonus-unit3 for:
  - gymnasium 1.x + mujoco 3.x (replaces gym 0.21 + mujoco_py)
  - HalfCheetah-v5 + Minari expert-v0 dataset (replaces D4RL v2 pkl)
  - gymnasium.wrappers.RecordVideo (replaces colabgymrender)
"""
import os
import random
import argparse
from dataclasses import dataclass

import numpy as np
import torch
import mujoco
import gymnasium as gym
import minari
from transformers import (
    DecisionTransformerConfig,
    DecisionTransformerModel,
    Trainer,
    TrainingArguments,
)

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=6000, help="number of training epochs")
parser.add_argument("--output_dir", type=str, default="output/", help="checkpoint output directory")
parser.add_argument("--save_total_limit", type=int, default=3, help="max checkpoints to keep")
parser.add_argument("--eval_only", action="store_true", help="skip training, only run evaluation")
parser.add_argument("--render", action="store_true", help="show live GUI window instead of saving video")
args = parser.parse_args()

os.environ["WANDB_DISABLED"] = "true"
# Offscreen rendering backend — egl uses the GPU (works on WSL2 with NVIDIA).
# Fall back to "osmesa" (CPU) if no GPU: sudo apt-get install libosmesa6-dev
os.environ.setdefault("MUJOCO_GL", "egl")


# ── Data ─────────────────────────────────────────────────────────────────────────

_minari_ds = minari.load_dataset("mujoco/halfcheetah/expert-v0", download=True)
dataset = [
    {
        "observations": ep.observations[:-1],          # (T, 17) align with actions
        "actions":      ep.actions,                    # (T, 6)
        "rewards":      ep.rewards,                    # (T,)
        "terminals":    ep.terminations | ep.truncations,  # (T,) bool
    }
    for ep in _minari_ds.iterate_episodes()
]


@dataclass
class DecisionTransformerGymDataCollator:
    return_tensors: str = "pt"
    max_len: int = 20
    state_dim: int = 17
    act_dim: int = 6
    max_ep_len: int = 1000
    scale: float = 1000.0
    state_mean: np.ndarray = None
    state_std: np.ndarray = None
    p_sample: np.ndarray = None
    n_traj: int = 0

    def __init__(self, dataset) -> None:
        self.act_dim = len(dataset[0]["actions"][0])
        self.state_dim = len(dataset[0]["observations"][0])
        self.dataset = dataset
        states = []
        traj_lens = []
        for traj in dataset:
            states.extend(traj["observations"])
            traj_lens.append(len(traj["observations"]))
        self.n_traj = len(traj_lens)
        states = np.vstack(states)
        self.state_mean = np.mean(states, axis=0)
        self.state_std = np.std(states, axis=0) + 1e-6
        traj_lens = np.array(traj_lens)
        self.p_sample = traj_lens / sum(traj_lens)

    def _discount_cumsum(self, x, gamma):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum

    def __call__(self, features):
        batch_size = len(features)
        batch_inds = np.random.choice(
            np.arange(self.n_traj),
            size=batch_size,
            replace=True,
            p=self.p_sample,
        )
        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for ind in batch_inds:
            feature = self.dataset[int(ind)]
            si = random.randint(0, len(feature["rewards"]) - 1)
            s.append(np.array(feature["observations"][si : si + self.max_len]).reshape(1, -1, self.state_dim))
            a.append(np.array(feature["actions"][si : si + self.max_len]).reshape(1, -1, self.act_dim))
            r.append(np.array(feature["rewards"][si : si + self.max_len]).reshape(1, -1, 1))
            d.append(np.array(feature["terminals"][si : si + self.max_len]).reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= self.max_ep_len] = self.max_ep_len - 1
            rtg.append(
                self._discount_cumsum(np.array(feature["rewards"][si:]), gamma=1.0)[
                    : s[-1].shape[1]
                ].reshape(1, -1, 1)
            )
            if rtg[-1].shape[1] < s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, self.state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - self.state_mean) / self.state_std
            a[-1] = np.concatenate([np.ones((1, self.max_len - tlen, self.act_dim)) * -10.0, a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, self.max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), rtg[-1]], axis=1) / self.scale
            timesteps[-1] = np.concatenate([np.zeros((1, self.max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, self.max_len - tlen)), np.ones((1, tlen))], axis=1))
        s = torch.from_numpy(np.concatenate(s, axis=0)).float()
        a = torch.from_numpy(np.concatenate(a, axis=0)).float()
        r = torch.from_numpy(np.concatenate(r, axis=0)).float()
        d = torch.from_numpy(np.concatenate(d, axis=0))
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).float()
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).long()
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).float()
        return {
            "states": s,
            "actions": a,
            "rewards": r,
            "returns_to_go": rtg,
            "timesteps": timesteps,
            "attention_mask": mask,
        }


# ── Model ─────────────────────────────────────────────────────────────────────────

class TrainableDT(DecisionTransformerModel):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        action_preds = output[1]
        action_targets = kwargs["actions"]
        attention_mask = kwargs["attention_mask"]
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_targets = action_targets.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        loss = torch.mean((action_preds - action_targets) ** 2)
        return {"loss": loss}

    def original_forward(self, **kwargs):
        return super().forward(**kwargs)


# ── Training ──────────────────────────────────────────────────────────────────────

CONTEXT_LEN = 20  # context window length (max_length was removed from config in transformers 5.x)

collator = DecisionTransformerGymDataCollator(dataset)
config = DecisionTransformerConfig(state_dim=collator.state_dim, act_dim=collator.act_dim)
model = TrainableDT(config)

training_args = TrainingArguments(
    output_dir=args.output_dir,
    remove_unused_columns=False,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=64,
    learning_rate=1e-4,
    weight_decay=1e-4,
    warmup_ratio=0.1,
    optim="adamw_torch",
    max_grad_norm=0.25,
    save_total_limit=args.save_total_limit,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collator,
)

if not args.eval_only:
    trainer.train()


# ── Evaluation ────────────────────────────────────────────────────────────────────

def get_action(model, states, actions, rewards, returns_to_go, timesteps):
    states = states.reshape(1, -1, model.config.state_dim)
    actions = actions.reshape(1, -1, model.config.act_dim)
    returns_to_go = returns_to_go.reshape(1, -1, 1)
    timesteps = timesteps.reshape(1, -1)

    states = states[:, -CONTEXT_LEN :]
    actions = actions[:, -CONTEXT_LEN :]
    returns_to_go = returns_to_go[:, -CONTEXT_LEN :]
    timesteps = timesteps[:, -CONTEXT_LEN :]
    padding = CONTEXT_LEN - states.shape[1]

    dev = states.device
    attention_mask = torch.cat([torch.zeros(padding, device=dev), torch.ones(states.shape[1], device=dev)])
    attention_mask = attention_mask.to(dtype=torch.long).reshape(1, -1)
    states = torch.cat([torch.zeros((1, padding, model.config.state_dim), device=dev), states], dim=1).float()
    actions = torch.cat([torch.zeros((1, padding, model.config.act_dim), device=dev), actions], dim=1).float()
    returns_to_go = torch.cat([torch.zeros((1, padding, 1), device=dev), returns_to_go], dim=1).float()
    timesteps = torch.cat([torch.zeros((1, padding), dtype=torch.long, device=dev), timesteps], dim=1)

    state_preds, action_preds, return_preds = model.original_forward(
        states=states,
        actions=actions,
        rewards=rewards,
        returns_to_go=returns_to_go,
        timesteps=timesteps,
        attention_mask=attention_mask,
        return_dict=False,
    )
    return action_preds[0, -1]


device = "cuda" if torch.cuda.is_available() else "cpu"
if args.eval_only:
    # load latest checkpoint from output_dir
    import glob, os as _os
    ckpts = sorted(glob.glob(f"{args.output_dir}/checkpoint-*"),
                   key=lambda p: int(p.split("-")[-1]))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {args.output_dir}")
    model = TrainableDT.from_pretrained(ckpts[-1])
    print(f"Loaded checkpoint: {ckpts[-1]}")
model = model.to(device)
scale = 1000.0
TARGET_RETURN = 16000 / scale  # Minari expert-v0 mean return ~16500
max_ep_len = 1000

state_mean = torch.from_numpy(collator.state_mean.astype(np.float32)).to(device)
state_std = torch.from_numpy(collator.state_std.astype(np.float32)).to(device)

if args.render:
    os.environ["MUJOCO_GL"] = "glfw"
    env = gym.make("HalfCheetah-v5", render_mode="human")
else:
    env = gym.make("HalfCheetah-v5", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, video_folder="./video", episode_trigger=lambda ep: True)

state_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

episode_return, episode_length = 0, 0
# gymnasium 0.26+: reset() returns (obs, info)
state, _ = env.reset()
if args.render:
    env.render()  # init viewer
    cam = env.unwrapped.mujoco_renderer.viewer.cam
    cam.trackbodyid = 1                              # track torso
    cam.type = int(mujoco.mjtCamera.mjCAMERA_TRACKING)
    cam.distance = 4.0
    cam.elevation = -20
target_return = torch.tensor(TARGET_RETURN, device=device, dtype=torch.float32).reshape(1, 1)
states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
rewards = torch.zeros(0, device=device, dtype=torch.float32)
timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

for t in range(max_ep_len):
    actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
    rewards = torch.cat([rewards, torch.zeros(1, device=device)])

    action = get_action(
        model,
        (states - state_mean) / state_std,
        actions,
        rewards,
        target_return,
        timesteps,
    )
    actions[-1] = action
    action = action.detach().cpu().numpy()

    # gymnasium 0.26+: step() returns (obs, reward, terminated, truncated, info)
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
    states = torch.cat([states, cur_state], dim=0)
    rewards[-1] = reward

    pred_return = target_return[0, -1] - (reward / scale)
    target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
    timesteps = torch.cat(
        [timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1
    )

    episode_return += reward
    episode_length += 1

    if done:
        break

env.close()
print(f"Episode return: {episode_return:.2f}, length: {episode_length}")
if not args.render:
    print("Video saved to ./video/")
