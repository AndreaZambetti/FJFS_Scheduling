"""Mini demo: deterministic shortest-duration heuristic on a 3x2 FJSP."""

import os
import sys

import numpy as np

try:  # Optional plotting
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
except ImportError:  # pragma: no cover - plotting optional
    plt = None
    Patch = None


ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)
PKG_ROOT = os.path.join(ROOT, "FJSP_RealWorld")
if PKG_ROOT not in sys.path:
    sys.path.append(PKG_ROOT)

from FJSP_RealWorld.FJSP_Env import FJSP


def main():
    np.random.seed(0)

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

    env = FJSP(n_j=n_jobs, n_m=n_machines, EachJob_num_operation=ops_per_job)

    (adj, features, omega, mask_job,
     mask_mch, dur, mch_time, job_time) = env.reset(durations)

    omega = np.asarray(omega)
    mask_job = np.asarray(mask_job)
    mask_mch = np.asarray(mask_mch)
    if mask_mch.ndim == 3:
        mask_mch = mask_mch[0]
    first_col = np.asarray(env.first_col[0])

    print("mask_mch shape:", mask_mch.shape)

    print("Initial omega per job:", omega)
    print("Initial job masks:", mask_job)

    step = 0
    done = False
    while not done:
        feasible_jobs = np.where(mask_job[0] == 0)[0]

        best_job = None
        best_value = float("inf")
        for job_id in feasible_jobs:
            action_candidate = int(omega[0, job_id])
            machines = np.where(mask_mch[action_candidate] == 0)[0]
            if machines.size == 0:
                continue
            op_idx = action_candidate - first_col[job_id]
            op_durations = durations[0, job_id, op_idx]
            feasible_times = op_durations[machines]
            avg_time = feasible_times.mean()
            if avg_time < best_value:
                best_value = avg_time
                best_job = job_id
                best_action = action_candidate
                best_machine = machines[np.argmin(feasible_times)]

        if best_job is None:
            raise RuntimeError("No feasible job found by the heuristic.")

        job_idx = np.searchsorted(first_col, best_action, side="right") - 1
        op_idx = best_action - first_col[job_idx]

        print(f"\nStep {step + 1}: scheduling job {job_idx + 1} op {op_idx + 1} on machine {best_machine + 1}")

        (adj, features, reward, done_flags, omega, mask_job,
         _, mask_mch, mch_time, job_time) = env.step(
            np.array([best_action]), np.array([best_machine])
        )

        omega = np.asarray(omega)
        mask_job = np.asarray(mask_job)
        mask_mch = np.asarray(mask_mch)
        if mask_mch.ndim == 3:
            mask_mch = mask_mch[0]
        done = bool(done_flags[0])
        step += 1

        print("Reward:", reward[0])
        print("Updated omega:", omega)
        print("Job masks:", mask_job)

    makespan = env.mchsEndTimes.max(-1).max(-1)
    print("\nEpisode finished in", step, "steps")
    print("Makespan:", makespan)

    if plt is None:
        print("matplotlib non disponibile: salto il Gantt.")
        return

    starts = env.mchsStartTimes[0]
    ends = env.mchsEndTimes[0]
    op_ids = env.opIDsOnMchs[0]

    fig, ax = plt.subplots(figsize=(7, 3))
    cmap = plt.cm.tab10
    legend_colors = {}

    for mch_idx in range(n_machines):
        for start, end, op_id in zip(starts[mch_idx], ends[mch_idx], op_ids[mch_idx]):
            if start < 0 or end < 0 or op_id < 0:
                continue
            job_idx = np.searchsorted(first_col, op_id, side="right") - 1
            op_idx = op_id - first_col[job_idx]
            color = cmap(job_idx % 10)
            ax.barh(
                mch_idx,
                width=end - start,
                left=start,
                height=0.4,
                color=color,
                edgecolor="black",
            )
            ax.text(
                start + (end - start) / 2,
                mch_idx,
                f"J{job_idx + 1}-O{op_idx + 1}",
                va="center",
                ha="center",
                fontsize=8,
                color="white",
            )
            legend_colors.setdefault(job_idx + 1, color)

    ax.set_yticks(range(n_machines))
    ax.set_yticklabels([f"Machine {i + 1}" for i in range(n_machines)])
    ax.set_xlabel("Time")
    ax.set_title(f"Heuristic schedule (makespan {makespan[0]:.1f})")

    handles = [Patch(color=color, label=f"Job {job}") for job, color in legend_colors.items()]
    if handles:
        ax.legend(handles=handles, loc="upper right")

    plt.tight_layout()
    output_path = os.path.join(ROOT, "mini_rl_gantt.png")
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print("Gantt salvato in:", output_path)


if __name__ == "__main__":
    main()
