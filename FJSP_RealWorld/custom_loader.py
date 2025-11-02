import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple


def load_microlotti_instance(json_path: str, batch_size: int = 1) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Convert a microlotti JSON description into the tensor format expected by FJSP_Env1.

    Returns
    -------
    durations : np.ndarray
        Shape (batch_size, n_jobs, max_ops, n_machines) with processing times in minutes.
        Each lot is represented as a single operation; value 0 means machine/tool incompatibile.
    ops_per_job : np.ndarray
        Shape (1, n_jobs) with the number of operations per job (all ones in this scenario).
    metadata : Dict
        Echoes the input data (machines, tools, lots) and includes the computed processing times.
    """

    path = Path(json_path)
    data = json.loads(path.read_text())

    machines = data["machines"]
    tools = data["tools"]
    lots = data["lots"]

    n_jobs = len(lots)
    n_machines = len(machines)

    # Every microlotto is a single operation in this first model
    max_ops = 1
    durations = np.zeros((batch_size, n_jobs, max_ops, n_machines), dtype=np.float32)

    processed_info = []
    for job_idx, lot in enumerate(lots):
        tool_id = lot["tool"]
        tool_cfg = tools[tool_id]
        setup = tool_cfg["setup_minutes"]
        per_piece = tool_cfg["process_minutes_per_piece"]
        pieces = lot["pieces"]
        total_time = setup + per_piece * pieces

        for machine_idx in range(n_machines):
            durations[:, job_idx, 0, machine_idx] = total_time

        processed_info.append(
            {
                "lot": lot["id"],
                "tool": tool_id,
                "setup": setup,
                "per_piece": per_piece,
                "pieces": pieces,
                "total_time": total_time,
            }
        )

    ops_per_job = np.ones((1, n_jobs), dtype=int)

    metadata = {
        "machines": machines,
        "tools": tools,
        "lots": lots,
        "processed": processed_info,
    }

    return durations, ops_per_job, metadata
