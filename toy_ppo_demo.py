"""Toy PPO micro-training on a 3x2 instance (LBM + finished mark features)."""

import os
import sys
from copy import deepcopy

import numpy as np
import torch


ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)
PKG_ROOT = os.path.join(ROOT, "FJSP_RealWorld")
if PKG_ROOT not in sys.path:
    sys.path.append(PKG_ROOT)

from Params import configs  # noqa: E402
from FJSP_RealWorld.FJSP_Env import FJSP  # noqa: E402
import FJSP_RealWorld.PPOwithValue as ppo_mod  # noqa: E402
from FJSP_RealWorld.PPOwithValue import PPO  # noqa: E402
from FJSP_RealWorld.mb_agg import aggr_obs, g_pool_cal  # noqa: E402


def build_toy_tensor():
    n_jobs = 3
    n_machines = 2
    ops_per_job = np.array([[2, 2, 2]], dtype=int)

    durations = np.zeros((1, n_jobs, ops_per_job.max(), n_machines), dtype=np.float32)
    durations[0, 0, 0] = [5.0, 3.0]
    durations[0, 0, 1] = [4.0, 0.0]
    durations[0, 1, 0] = [6.0, 2.0]
    durations[0, 1, 1] = [0.0, 4.5]
    durations[0, 2, 0] = [3.2, 4.0]
    durations[0, 2, 1] = [5.0, 1.0]

    return ops_per_job, durations


def rollout(agent, ops_per_job, durations, greedy):
    env = FJSP(n_j=3, n_m=2, EachJob_num_operation=ops_per_job)
    (
        adj, features, candidate, mask_job,
        mask_mch, dur, mch_time, job_time,
    ) = env.reset(durations)

    device = torch.device(configs.device)
    num_tasks = env.number_of_tasks
    g_pool_step = g_pool_cal(
        graph_pool_type=configs.graph_pool_type,
        batch_size=torch.Size([1, num_tasks, num_tasks]),
        n_nodes=num_tasks,
        device=device,
    )

    pool = None
    step = 0
    done = False
    rewards = []
    job_logs, mch_logs = [], []

    while not done and step < 20:
        env_adj = aggr_obs(deepcopy(adj).to(device).to_sparse(), num_tasks)
        env_fea = torch.from_numpy(np.copy(features)).float().to(device)
        env_fea = env_fea.reshape(-1, env_fea.size(-1))
        env_candidate = torch.from_numpy(np.copy(candidate)).long().to(device)
        env_mask = torch.from_numpy(np.copy(mask_job)).to(device)
        if env_candidate.dim() == 1:
            env_candidate = env_candidate.unsqueeze(0)
        if env_mask.dim() == 1:
            env_mask = env_mask.unsqueeze(0)
        env_mask_mch = torch.from_numpy(np.copy(mask_mch)).to(device)
        if env_mask_mch.dim() == 2:
            env_mask_mch = env_mask_mch.unsqueeze(0)
        env_dur = torch.from_numpy(np.copy(dur)).float().to(device)
        env_mch_time = torch.from_numpy(np.copy(mch_time)).float().to(device)

        action, _, log_a, action_node, _, mask_mch_action, hx = agent.policy_job(
            x=env_fea,
            graph_pool=g_pool_step,
            padded_nei=None,
            adj=env_adj,
            candidate=env_candidate,
            mask=env_mask,
            mask_mch=env_mask_mch,
            dur=env_dur,
            a_index=0,
            old_action=0,
            mch_pool=pool,
            old_policy=True,
            T=1,
            greedy=greedy,
        )

        if action_node.dim() == 1:
            action_node = action_node.unsqueeze(0)
        if env_mch_time.dim() == 1:
            env_mch_time = env_mch_time.unsqueeze(0)

        pi_mch, _, pool = agent.policy_mch(action_node, hx, mask_mch_action, env_mch_time)
        if greedy:
            _, mch_choice = pi_mch.squeeze(-1).max(1)
            log_mch = torch.zeros(1, device=device)
        else:
            dist = torch.distributions.Categorical(pi_mch.squeeze())
            mch_choice = dist.sample()
            log_mch = dist.log_prob(mch_choice)

        action_id = int(action.item() if action.dim() == 1 else action.cpu().numpy()[0])
        machine_id = int(mch_choice.item())

        (
            adj, features, reward, done_flags, candidate, mask_job,
            _, mask_mch, mch_time, job_time,
        ) = env.step(
            np.array([action_id], dtype=int),
            np.array([machine_id], dtype=int),
        )

        rewards.append(torch.tensor(reward[0], dtype=torch.float32, device=device))
        if not greedy:
            job_logs.append(log_a.squeeze())
            mch_logs.append(log_mch.squeeze())

        candidate = np.asarray(candidate)
        if candidate.ndim == 1:
            candidate = candidate[np.newaxis, :]
        mask_job = np.asarray(mask_job)
        if mask_job.ndim == 1:
            mask_job = mask_job[np.newaxis, :]
        mask_mch = np.asarray(mask_mch)
        if mask_mch.ndim == 2:
            mask_mch = mask_mch[np.newaxis, :, :]

        done = bool(done_flags[0])
        step += 1

    return {
        "rewards": rewards,
        "job_logs": job_logs,
        "mch_logs": mch_logs,
        "steps": step,
        "env": env,
        "features": features,
        "mask_job": mask_job,
    }


def train_micro(agent, ops_per_job, durations, epochs=20, batch_size=4):
    print("\n== Micro-training (REINFORCE) ==")
    device = torch.device(configs.device)
    gamma = configs.gamma
    for epoch in range(epochs):
        all_job_logs, all_mch_logs, all_returns = [], [], []
        total_reward = 0.0

        for _ in range(batch_size):
            data = rollout(agent, ops_per_job, durations, greedy=False)
            g = torch.tensor(0.0, device=device)
            returns = []
            for r in reversed(data["rewards"]):
                g = r + gamma * g
                returns.insert(0, g)
            returns = torch.stack(returns)

            all_job_logs.extend(data["job_logs"])
            all_mch_logs.extend(data["mch_logs"])
            all_returns.extend(returns)
            total_reward += sum(r.item() for r in data["rewards"])

        returns_tensor = torch.stack(all_returns)
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)

        agent.job_optimizer.zero_grad()
        agent.mch_optimizer.zero_grad()
        job_loss = -(torch.stack(all_job_logs) * returns_tensor).mean()
        mch_loss = -(torch.stack(all_mch_logs) * returns_tensor).mean()
        loss = job_loss + mch_loss
        loss.backward()
        agent.job_optimizer.step()
        agent.mch_optimizer.step()

        avg_reward = total_reward / batch_size
        print(f"Epoch {epoch + 1:02d} | avg reward {avg_reward:5.2f} | loss {loss.item():.4f}")


def showcase(agent, ops_per_job, durations):
    env = FJSP(n_j=3, n_m=2, EachJob_num_operation=ops_per_job)
    (
        adj, features, candidate, mask_job,
        mask_mch, dur, mch_time, job_time,
    ) = env.reset(durations)

    device = torch.device(configs.device)
    num_tasks = env.number_of_tasks
    g_pool_step = g_pool_cal(
        graph_pool_type=configs.graph_pool_type,
        batch_size=torch.Size([1, num_tasks, num_tasks]),
        n_nodes=num_tasks,
        device=device,
    )

    pool = None
    step = 0
    done = False
    while not done and step < 20:
        env_adj = aggr_obs(deepcopy(adj).to(device).to_sparse(), num_tasks)
        env_fea = torch.from_numpy(np.copy(features)).float().to(device)
        env_fea = env_fea.reshape(-1, env_fea.size(-1))
        env_candidate = torch.from_numpy(np.copy(candidate)).long().to(device)
        env_mask = torch.from_numpy(np.copy(mask_job)).to(device)
        env_mask_mch = torch.from_numpy(np.copy(mask_mch)).to(device)
        env_dur = torch.from_numpy(np.copy(dur)).float().to(device)
        env_mch_time = torch.from_numpy(np.copy(mch_time)).float().to(device)

        if env_candidate.dim() == 1:
            env_candidate = env_candidate.unsqueeze(0)
        if env_mask.dim() == 1:
            env_mask = env_mask.unsqueeze(0)
        if env_mask_mch.dim() == 2:
            env_mask_mch = env_mask_mch.unsqueeze(0)

        action, _, _, action_node, _, mask_mch_action, hx = agent.policy_job(
            x=env_fea,
            graph_pool=g_pool_step,
            padded_nei=None,
            adj=env_adj,
            candidate=env_candidate,
            mask=env_mask,
            mask_mch=env_mask_mch,
            dur=env_dur,
            a_index=0,
            old_action=0,
            mch_pool=pool,
            old_policy=True,
            T=1,
            greedy=True,
        )

        if action_node.dim() == 1:
            action_node = action_node.unsqueeze(0)
        if env_mch_time.dim() == 1:
            env_mch_time = env_mch_time.unsqueeze(0)

        pi_mch, _, pool = agent.policy_mch(action_node, hx, mask_mch_action, env_mch_time)
        _, mch_choice = pi_mch.squeeze(-1).max(1)

        action_id = int(action.item() if action.dim() == 1 else action.cpu().numpy()[0])
        machine_id = int(mch_choice.item())

        print(f"\nStep {step + 1}: action {action_id} on machine {machine_id}")

        (
            adj, features, reward, done_flags, candidate, mask_job,
            _, mask_mch, mch_time, job_time,
        ) = env.step(
            np.array([action_id], dtype=int),
            np.array([machine_id], dtype=int),
        )

        candidate = np.asarray(candidate)
        if candidate.ndim == 1:
            candidate = candidate[np.newaxis, :]
        mask_job = np.asarray(mask_job)
        if mask_job.ndim == 1:
            mask_job = mask_job[np.newaxis, :]
        mask_mch = np.asarray(mask_mch)
        if mask_mch.ndim == 2:
            mask_mch = mask_mch[np.newaxis, :, :]

        flat_features = features.reshape(-1, features.shape[-1])
        print("Reward:", reward[0])
        print("Job flags (mask):", mask_job[0])
        print("Task finished (from features):", flat_features[:, 1].astype(int))
        print("LBM values:", flat_features[:, 0])

        done = bool(done_flags[0])
        step += 1

    print("\nMakespan LB:", env.LBm[0].max())


def main():
    configs.device = "cpu"
    configs.n_j = 3
    configs.n_m = 2
    ppo_mod.device = torch.device("cpu")

    ops_per_job, durations = build_toy_tensor()

    agent = PPO(
        lr=configs.lr,
        gamma=configs.gamma,
        k_epochs=configs.k_epochs,
        eps_clip=configs.eps_clip,
        n_j=3,
        n_m=2,
        num_layers=configs.num_layers,
        neighbor_pooling_type=configs.neighbor_pooling_type,
        input_dim=configs.input_dim,
        hidden_dim=configs.hidden_dim,
        num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
        num_mlp_layers_actor=configs.num_mlp_layers_actor,
        hidden_dim_actor=configs.hidden_dim_actor,
        num_mlp_layers_critic=configs.num_mlp_layers_critic,
        hidden_dim_critic=configs.hidden_dim_critic,
    )

    pre_data = rollout(agent, ops_per_job, durations, greedy=True)
    flat_features = pre_data["features"].reshape(-1, pre_data["features"].shape[-1])
    base_reward = sum(r.item() for r in pre_data["rewards"])
    base_lb = float(pre_data["env"].LBm[0].max())
    print("Feature tensor shape:", pre_data["features"].shape)
    print("First operations [LBM, finished_mark]:")
    print(flat_features)
    print(f"Baseline total reward: {base_reward:.2f} | makespan LB: {base_lb:.2f}")

    train_micro(agent, ops_per_job, durations)

    post_data = rollout(agent, ops_per_job, durations, greedy=True)
    post_reward = sum(r.item() for r in post_data["rewards"])
    post_lb = float(post_data["env"].LBm[0].max())
    print(f"\nAfter training total reward: {post_reward:.2f} | makespan LB: {post_lb:.2f}")

    print("\n== Greedy inference after training ==")
    showcase(agent, ops_per_job, durations)


if __name__ == "__main__":
    main()
