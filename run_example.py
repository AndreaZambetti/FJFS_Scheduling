"""Benchmark roll-out using pretrained PPO policies on HurinkVdata39."""

from __future__ import annotations

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

# Use CPU unless CUDA is available.
configs.device = 'cuda' if torch.cuda.is_available() else 'cpu'

from FJSP_RealWorld.DataRead import getdata  # noqa: E402
from FJSP_RealWorld.FJSP_Env import FJSP  # noqa: E402
from FJSP_RealWorld.PPOwithValue import PPO  # noqa: E402
from FJSP_RealWorld.mb_agg import aggr_obs, g_pool_cal  # noqa: E402


def build_instance_tensor(instance_path: str):
    """Convert a .fjs file into the tensor format expected by FJSP.reset."""

    raw = getdata(instance_path)

    n_jobs = raw['n']
    n_machines = raw['m']

    ops_per_job = np.array([[len(raw['OJ'][job]) for job in sorted(raw['OJ'])]], dtype=int)

    max_ops = ops_per_job.max()
    durations = np.zeros((1, n_jobs, max_ops, n_machines), dtype=np.float32)

    for (job, op), machines in raw['operations_machines'].items():
        for mach in machines:
            durations[0, job - 1, op - 1, mach - 1] = raw['operations_times'][(job, op, mach)]

    return raw, ops_per_job, durations


def decode_action(first_col: np.ndarray, last_col: np.ndarray, action_id: int):
    for job_idx, (start, end) in enumerate(zip(first_col, last_col)):
        if start <= action_id <= end:
            return job_idx + 1, action_id - start + 1
    raise ValueError(f"Action {action_id} is outside the job range.")


def pick_feasible_machine(mask_array, action_id: int) -> int:
    mask_array = np.asarray(mask_array)
    if mask_array.ndim == 3:
        machine_mask = mask_array[0, action_id]
    elif mask_array.ndim == 2:
        machine_mask = mask_array[action_id]
    else:
        raise ValueError(f"Unexpected mask shape {mask_array.shape}.")
    selectable = np.where(machine_mask == 0)[0]
    if selectable.size == 0:
        raise RuntimeError(f"No feasible machines for action {action_id}.")
    return int(selectable[0])


def load_policies(n_jobs: int, n_machines: int, checkpoint_root: str) -> PPO:
    # Align global config with current instance dimensions.
    configs.n_j = n_jobs
    configs.n_m = n_machines

    agent = PPO(
        lr=configs.lr,
        gamma=configs.gamma,
        k_epochs=configs.k_epochs,
        eps_clip=configs.eps_clip,
        n_j=n_jobs,
        n_m=n_machines,
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

    job_path = os.path.join(checkpoint_root, "policy_job.pth")
    mch_path = os.path.join(checkpoint_root, "policy_mch.pth")

    device = torch.device(configs.device)
    agent.policy_job.load_state_dict(torch.load(job_path, map_location=device))
    agent.policy_mch.load_state_dict(torch.load(mch_path, map_location=device))

    agent.policy_job.eval()
    agent.policy_mch.eval()
    return agent


def main():
    instance_path = os.path.join(
        ROOT, "FJSP_RealWorld", "FJSSPinstances", "M15", "HurinkVdata39.fjs"
    )

    raw, ops_per_job, durations = build_instance_tensor(instance_path)

    env = FJSP(n_j=raw['n'], n_m=raw['m'], EachJob_num_operation=ops_per_job)

    (
        adj, features, candidate, mask_job,
        mask_mch, dur, mch_time, job_time,
    ) = env.reset(durations)

    number_of_tasks = env.number_of_tasks
    g_pool_step = g_pool_cal(
        graph_pool_type=configs.graph_pool_type,
        batch_size=torch.Size([1, number_of_tasks, number_of_tasks]),
        n_nodes=number_of_tasks,
        device=torch.device(configs.device),
    )

    agent = load_policies(
        n_jobs=raw['n'],
        n_machines=raw['m'],
        checkpoint_root=os.path.join(
            ROOT,
            "FJSP_RealWorld",
            "saved_network",
            "FJSP_J15M15",
            "best_value0",
        ),
    )

    first_col = np.asarray(env.first_col[0])
    last_col = np.asarray(env.last_col[0])

    candidate = np.asarray(candidate)
    mask_job = np.asarray(mask_job)
    mask_mch = np.asarray(mask_mch)

    print("=== Initial observation ===")
    print("Ready operations per job (omega):", candidate)
    print("Job finished flags:", mask_job)
    print("Feature tensor shape:", features.shape)

    device = torch.device(configs.device)
    pool = None
    step = 0
    rollout_log = []

    while True:
        env_adj = aggr_obs(deepcopy(adj).to(device).to_sparse(), number_of_tasks)
        env_fea = torch.from_numpy(np.copy(features)).float().to(device)
        env_fea = env_fea.reshape(-1, env_fea.size(-1))
        env_candidate = torch.from_numpy(np.copy(candidate)).long().to(device)
        env_mask = torch.from_numpy(np.copy(mask_job)).to(device)
        env_mask_mch = torch.from_numpy(np.copy(mask_mch)).to(device)
        env_dur = torch.from_numpy(np.copy(dur)).float().to(device)
        env_mch_time = torch.from_numpy(np.copy(mch_time)).float().to(device)

        action, a_idx, _, action_node, _, mask_mch_action, hx = agent.policy_job(
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

        action_id = int(action.cpu().numpy()[0])
        machine_id = int(mch_choice.cpu().numpy()[0])
        job_id, op_id = decode_action(first_col, last_col, action_id)

        (
            adj, features, rewards, done, candidate, mask_job,
            _, mask_mch, mch_time, job_time,
        ) = env.step(action.cpu().numpy(), mch_choice.cpu().numpy())

        candidate = np.asarray(candidate)
        mask_job = np.asarray(mask_job)
        mask_mch = np.asarray(mask_mch)
        done = np.asarray(done)
        if hasattr(env, "op_dur"):
            dur = np.asarray(env.op_dur)
        else:
            dur = np.asarray(env.dur)

        step += 1
        rollout_log.append(
            {
                "step": step,
                "job": job_id,
                "op": op_id,
                "action": action_id,
                "machine": machine_id + 1,
                "reward": float(rewards[0]),
                "lbm": float(env.LBm[0].max()),
                "partial_len": int(np.count_nonzero(env.partial_sol_sequeence[0] >= 0)),
            }
        )

        loggable = step <= 10 or step % 25 == 0 or done[0]
        if loggable:
            print(f"\n-- Dispatch {step} --")
            print(
                f"Policy selected job {job_id}, operation {op_id} (action {action_id})"
            )
            print(f"Machine according to policy: {machine_id + 1}")
            print(f"Reward: {rewards[0]:.3f}")
            print("Updated lower-bound makespan:", env.LBm[0].max())
            print("Omega head:", candidate[:, :min(5, candidate.shape[1])])
            print(
                "Partial sequence length:",
                np.count_nonzero(env.partial_sol_sequeence[0] >= 0),
            )

        if done[0]:
            break

    makespan = env.mchsEndTimes.max(-1).max(-1)
    print("\n=== Episode summary ===")
    print("Total dispatches:", step)
    print("Final makespan (policy):", makespan)
    print("Aggregated positive rewards:", env.posRewards[0])

    tail_events = rollout_log[-5:]
    print("Last 5 dispatches:")
    for entry in tail_events:
        print(
            f"  step {entry['step']:>3}: job {entry['job']:>2} op {entry['op']:>2} "
            f"-> machine {entry['machine']:>2}, makespan LB {entry['lbm']:.1f}"
        )


if __name__ == "__main__":
    main()
