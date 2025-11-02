"""Demo di integrazione per i microlotti: carica il JSON e simula una schedulazione semplice.

L'obiettivo dello script è mostrare come collegare i dati custom (`custom_loader`)
all'ambiente `FJSP_RealWorld.FJSP`. Per chiarezza ogni passaggio è commentato.
"""

from __future__ import annotations

import os
import sys
from typing import List

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)
PKG_ROOT = os.path.join(ROOT, "FJSP_RealWorld")
if PKG_ROOT not in sys.path:
    sys.path.append(PKG_ROOT)

from Params import configs  # noqa: E402
from FJSP_RealWorld.FJSP_Env1 import FJSP  # noqa: E402
from FJSP_RealWorld.custom_loader import load_microlotti_instance  # noqa: E402
from FJSP_RealWorld.PPOwithValue import PPO  # noqa: E402
from FJSP_RealWorld.mb_agg import aggr_obs, g_pool_cal  # noqa: E402


def select_next_lot(
    omega: np.ndarray,
    mask_job: np.ndarray,
    machine_finish_times: np.ndarray,
    lot_priorities: List[float],
) -> tuple[int, int]:
    """Seleziona il prossimo microlotto e la macchina usando una logica basata sul makespan atteso.

    - Scegliamo tra i job non completati quello con tempo totale maggiore (priorità alta).
    - Assegniamo la macchina che si libera prima (minimo finish time attuale).
    """

    ready_jobs = np.where(mask_job[0] == 0)[0]
    if ready_jobs.size == 0:
        raise RuntimeError("Nessun job disponibile nonostante l'episodio non sia terminato.")

    # Ordina i job per tempo totale (desc) usando le priorità calcolate dal loader
    sorted_jobs = sorted(ready_jobs, key=lambda idx: lot_priorities[idx], reverse=True)

    job_idx = int(sorted_jobs[0])
    action_id = int(omega[0, job_idx])

    # Tempi di fine correnti: valori negativi indicano macchina inutilizzata → consideriamoli 0
    finish_times = machine_finish_times.copy()
    finish_times[finish_times < 0] = 0
    machine_choice = int(np.argmin(finish_times))

    return action_id, machine_choice


def main() -> None:
    # 1. Carichiamo i microlotti dal JSON (setup + tempi al pezzo)
    data_path = os.path.join(PKG_ROOT, "data", "microlotti_example.json")
    durations, ops_per_job, meta = load_microlotti_instance(data_path, batch_size=1)

    # 2. Allineiamo i parametri globali (n° job/macchine) al nuovo scenario
    configs.n_j = durations.shape[1]
    configs.n_m = durations.shape[-1]

    # 3. Creiamo l'ambiente usando il vettore "ops_per_job"
    env = FJSP(n_j=configs.n_j, n_m=configs.n_m, EachJob_num_operation=ops_per_job)

    # 4. Reset dell'ambiente: otteniamo stato iniziale (grafo, feature, maschere, ecc.)
    (
        adj,
        features,
        omega,
        mask_job,
        mask_mch,
        dur,
        mch_time,
        job_time,
    ) = env.reset(durations)

    # Iniettiamo informazioni sulle attrezzature per ogni operazione (richieste per il setup)
    # Mappiamo gli strumenti in un indice intero stabile
    tool_names = list(meta["tools"].keys())
    tool_index = {name: idx for idx, name in enumerate(tool_names)}
    tool_setup = np.array([meta["tools"][name]["setup_minutes"] for name in tool_names], dtype=np.float32)

    # Costruiamo `op_tools`: per il nostro esempio ogni job ha una sola operazione
    op_tools = -1 * np.ones((durations.shape[0], configs.n_j, ops_per_job.max()), dtype=int)
    for j, lot in enumerate(meta["lots"]):
        op_tools[0, j, 0] = tool_index[lot["tool"]]

    # Assegna all'ambiente i vettori necessari per attivare il setup dinamico
    env.op_tools = op_tools
    env.tool_setup_minutes = tool_setup

    # 4.b Inizializziamo l'agente PPO con pesi casuali (nessun checkpoint caricato)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PPO(
        lr=configs.lr,
        gamma=configs.gamma,
        k_epochs=configs.k_epochs,
        eps_clip=configs.eps_clip,
        n_j=configs.n_j,
        n_m=configs.n_m,
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
    agent.policy_job.to(device).eval()
    agent.policy_mch.to(device).eval()

    g_pool_step = g_pool_cal(
        graph_pool_type=configs.graph_pool_type,
        batch_size=torch.Size([1, env.number_of_tasks, env.number_of_tasks]),
        n_nodes=env.number_of_tasks,
        device=device,
    )

    print("=== Rollout PPO (pesi casuali, nessun training) ===")
    step_ppo = 0
    adj_ppo = adj.clone()
    fea_ppo = np.copy(features)
    omega_ppo = np.copy(omega)
    mask_job_ppo = np.copy(mask_job)
    mask_mch_ppo = np.copy(mask_mch)
    dur_ppo = np.copy(dur)
    mch_time_ppo = np.copy(mch_time)
    job_time_ppo = np.copy(job_time)

    while True:
        env_adj = aggr_obs(adj_ppo.to(device).to_sparse(), env.number_of_tasks)
        env_fea = torch.from_numpy(np.copy(fea_ppo)).float().to(device).reshape(-1, fea_ppo.shape[-1])
        env_candidate = torch.from_numpy(np.copy(omega_ppo)).long().to(device)
        env_mask = torch.from_numpy(np.copy(mask_job_ppo)).to(device)
        env_mask_mch = torch.from_numpy(np.copy(mask_mch_ppo)).to(device)
        env_dur = torch.from_numpy(np.copy(dur_ppo)).float().to(device)
        env_mch_time = torch.from_numpy(np.copy(mch_time_ppo)).float().to(device)

        action, a_idx, log_a, action_node, _, mask_mch_action, hx = agent.policy_job(
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
            mch_pool=None,
            old_policy=True,
            T=1,
            greedy=False,
        )

        if action.dim() == 0:
            action = action.unsqueeze(0)

        if action_node.dim() == 1:
            action_node = action_node.unsqueeze(0)
        if env_mch_time.dim() == 1:
            env_mch_time = env_mch_time.unsqueeze(0)
        if mask_mch_action.dim() == 2:
            mask_mch_action = mask_mch_action.unsqueeze(0)

        pi_mch, _, _ = agent.policy_mch(action_node, hx, mask_mch_action, env_mch_time)
        mch_dist = torch.distributions.Categorical(pi_mch.squeeze())
        mch_choice = mch_dist.sample().unsqueeze(0)

        (
            adj_ppo,
            fea_ppo,
            reward_ppo,
            done_ppo,
            omega_ppo,
            mask_job_ppo,
            _,
            mask_mch_ppo,
            mch_time_ppo,
            job_time_ppo,
        ) = env.step(action.cpu().numpy(), mch_choice.cpu().numpy())

        print(
            f"[PPO random] step {step_ppo}: azione {int(action.cpu().numpy()[0])}, "
            f"macchina {int(mch_choice.cpu().numpy()[0])}, reward {reward_ppo[0]:.2f}"
        )

        step_ppo += 1
        if done_ppo[0]:
            break

    print()
    random_makespan = float(env.mchsEndTimes.max(-1).max(-1)[0])

    # 4.c Proviamo a caricare un checkpoint pre-addestrato (se disponibile)
    checkpoint_root = os.path.join(
        PKG_ROOT, "saved_network", "FJSP_J15M15", "best_value0"
    )
    job_ckpt = os.path.join(checkpoint_root, "policy_job.pth")
    mch_ckpt = os.path.join(checkpoint_root, "policy_mch.pth")

    if os.path.exists(job_ckpt) and os.path.exists(mch_ckpt):
        agent.policy_job.load_state_dict(torch.load(job_ckpt, map_location=device))
        agent.policy_mch.load_state_dict(torch.load(mch_ckpt, map_location=device))
        agent.policy_job.eval()
        agent.policy_mch.eval()

        (
            adj_ckpt,
            features_ckpt,
            omega_ckpt,
            mask_job_ckpt,
            mask_mch_ckpt,
            dur_ckpt,
            mch_time_ckpt,
            job_time_ckpt,
        ) = env.reset(durations)

        print("=== Rollout PPO (checkpoint FJSP_J15M15/best_value0) ===")
        step_ckpt = 0
        while True:
            env_adj = aggr_obs(adj_ckpt.to(device).to_sparse(), env.number_of_tasks)
            env_fea = torch.from_numpy(np.copy(features_ckpt)).float().to(device).reshape(
                -1, features_ckpt.shape[-1]
            )
            env_candidate = torch.from_numpy(np.copy(omega_ckpt)).long().to(device)
            env_mask = torch.from_numpy(np.copy(mask_job_ckpt)).to(device)
            env_mask_mch = torch.from_numpy(np.copy(mask_mch_ckpt)).to(device)
            env_dur = torch.from_numpy(np.copy(dur_ckpt)).float().to(device)
            env_mch_time = torch.from_numpy(np.copy(mch_time_ckpt)).float().to(device)

            action, a_idx, log_a, action_node, _, mask_mch_action, hx = agent.policy_job(
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
                mch_pool=None,
                old_policy=True,
                T=1,
                greedy=True,
            )

            if action.dim() == 0:
                action = action.unsqueeze(0)
            if action_node.dim() == 1:
                action_node = action_node.unsqueeze(0)
            if env_mch_time.dim() == 1:
                env_mch_time = env_mch_time.unsqueeze(0)
            if mask_mch_action.dim() == 2:
                mask_mch_action = mask_mch_action.unsqueeze(0)

            pi_mch, _, _ = agent.policy_mch(action_node, hx, mask_mch_action, env_mch_time)
            _, mch_choice = pi_mch.squeeze(-1).max(1)

            (
                adj_ckpt,
                features_ckpt,
                reward_ckpt,
                done_ckpt,
                omega_ckpt,
                mask_job_ckpt,
                _,
                mask_mch_ckpt,
                mch_time_ckpt,
                job_time_ckpt,
            ) = env.step(action.cpu().numpy(), mch_choice.cpu().numpy())

            print(
                f"[PPO checkpoint] step {step_ckpt}: azione {int(action.cpu().numpy()[0])}, "
                f"macchina {int(mch_choice.cpu().numpy()[0])}, reward {reward_ckpt[0]:.2f}"
            )

            step_ckpt += 1
            if done_ckpt[0]:
                break
        print()
        checkpoint_makespan = float(env.mchsEndTimes.max(-1).max(-1)[0])
    else:
        print("=== Nessun checkpoint trovato: salto il rollout PPO pre-addestrato ===\n")
        checkpoint_makespan = None

    # Reinizializziamo l'ambiente per eseguire l'euristica base
    (
        adj,
        features,
        omega,
        mask_job,
        mask_mch,
        dur,
        mch_time,
        job_time,
    ) = env.reset(durations)

    print("=== Microlotti caricati ===")
    for info in meta["processed"]:
        print(
            f"Lot {info['lot']} usa {info['tool']} "
            f"(setup {info['setup']} min, {info['per_piece']} min/pezzo, "
            f"{info['pieces']} pezzi → totale {info['total_time']} min)"
        )
    print()

    first_col = np.asarray(env.first_col[0])
    machine_names = meta["machines"]

    lot_priorities = [info["total_time"] for info in meta["processed"]]
    step = 0
    timeline: List[str] = []

    while True:
        # 5. Selezioniamo l'azione con algoritmo "max total time + macchina più libera"
        machine_finish_times = env.mchsEndTimes[0].max(axis=1)
        action_id, machine_choice = select_next_lot(
            np.asarray(omega),
            np.asarray(mask_job),
            machine_finish_times,
            lot_priorities,
        )

        job_idx = int(np.searchsorted(first_col, action_id, side="right") - 1)
        timeline.append(
            f"Step {step}: pianifico operazione {action_id} del job {job_idx + 1} "
            f"su macchina {machine_names[machine_choice]}"
        )

        # 6. Avanziamo l'ambiente (convertendo in array di dimensione batch=1)
        (
            adj,
            features,
            reward,
            done,
            omega,
            mask_job,
            _,
            mask_mch,
            mch_time,
            job_time,
        ) = env.step(
            np.array([action_id], dtype=int),
            np.array([machine_choice], dtype=int),
        )

        print(
            f"Reward step {step}: {reward[0]:.2f} "
            f"– {machine_names[machine_choice]} termina a {env.mchsEndTimes[0, machine_choice].max():.1f} min"
        )

        step += 1
        if done[0]:
            break

    print("\n=== Timeline pianificazione (heuristica base) ===")
    print("\n".join(timeline))

    makespan = float(env.mchsEndTimes.max(-1).max(-1)[0])
    print(f"\nMakespan finale: {makespan:.1f} minuti")

    print("\n=== Riepilogo makespan ===")
    print(f"PPO random: {random_makespan:.1f} min")
    if checkpoint_makespan is not None:
        print(f"PPO checkpoint: {checkpoint_makespan:.1f} min")
    else:
        print("PPO checkpoint: n/d (checkpoint assente)")
    print(f"Euristica base: {makespan:.1f} min")


if __name__ == "__main__":
    main()
