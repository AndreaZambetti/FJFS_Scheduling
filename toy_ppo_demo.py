"""Toy rollout showing PPO inputs (LBM + finished mark) on a 3x2 instance."""

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
    durations[0, 1, 1] = [3.0, 4.5]
    durations[0, 2, 0] = [3.0, 4.0]
    durations[0, 2, 1] = [5.0, 1.0]

    return ops_per_job, durations


def main():
    configs.device = "cpu"
    configs.n_j = 3
    configs.n_m = 2
    ppo_mod.device = torch.device("cpu")

    ops_per_job, durations = build_toy_tensor()
    env = FJSP(n_j=3, n_m=2, EachJob_num_operation=ops_per_job)

    (
        adj, features, candidate, mask_job,
        mask_mch, dur, mch_time, job_time,
    ) = env.reset(durations)

    print("Feature tensor shape:", features.shape)
    flat_features = features.reshape(-1, features.shape[-1])
    print("First 6 operations [LBM, finished_mark]:")
    print(flat_features[:6])

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

    number_of_tasks = env.number_of_tasks
    pool = None
    device = torch.device(configs.device)
    g_pool_step = g_pool_cal(
        graph_pool_type=configs.graph_pool_type,
        batch_size=torch.Size([1, number_of_tasks, number_of_tasks]),
        n_nodes=number_of_tasks,
        device=device,
    )

    step = 0
    done = False
    while not done and step < 8:
        env_adj = aggr_obs(deepcopy(adj).to(device).to_sparse(), number_of_tasks)
        env_fea = torch.from_numpy(np.copy(features)).float().to(device)
        env_fea = env_fea.reshape(-1, env_fea.size(-1))
        env_candidate = torch.from_numpy(np.copy(candidate)).long().to(device)
        env_mask = torch.from_numpy(np.copy(mask_job)).to(device)
        env_mask_mch = torch.from_numpy(np.copy(mask_mch)).to(device)
        env_dur = torch.from_numpy(np.copy(dur)).float().to(device)
        env_mch_time = torch.from_numpy(np.copy(mch_time)).float().to(device)

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
        mch_choice = mch_choice.unsqueeze(0)

        action_id = int(action.cpu().numpy()[0])
        machine_id = int(mch_choice.cpu().numpy()[0])
        print(f"\nStep {step + 1}: policy picked action {action_id} on machine {machine_id}")

        (
            adj, features, rewards, done_flags, candidate, mask_job,
            _, mask_mch, mch_time, job_time,
        ) = env.step(
            np.array([action_id], dtype=int),
            np.array([machine_id], dtype=int),
        )

        print("Reward:", rewards[0])
        print("Updated finished marks:", mask_job[0])
        flat_features = features.reshape(-1, features.shape[-1])
        print("Sample features after step:", flat_features[:6])

        done = bool(done_flags[0])
        step += 1

    print("\nFinal LB matrix sample:")
    print(env.LBm[0][:6])
    print("Done in", step, "steps | makespan LB:", env.LBm[0].max())


if __name__ == "__main__":
    main()
