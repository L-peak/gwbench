"""Retrain the 30/90-day GRU models with a purged temporal split.

Kaggle script kernel. The panel and the updated training script are mounted
from the public gwbench-data dataset; outputs are written to /kaggle/working.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

INPUT = Path("/kaggle/input")
script = next(INPUT.rglob("scripts/07_train_gru.py"))
root = script.parents[1]
panel = next(INPUT.rglob("panel_core_clean.parquet"))
print(f"dataset root: {root}")
print(f"panel: {panel}")
print(f"training script: {script}")

runs = []
for horizon in (30, 90):
    for seed in (1, 2, 3):
        out_dir = Path(f"/kaggle/working/gru_purged_h{horizon}_seed{seed}")
        cmd = [
            sys.executable,
            str(script),
            "--panel",
            str(panel),
            "--horizon",
            str(horizon),
            "--lookback",
            "365",
            "--stride-train",
            "3",
            "--stride-eval",
            "1",
            "--epochs",
            "40",
            "--batch-size",
            "256",
            "--lr",
            "0.001",
            "--hidden",
            "64",
            "--layers",
            "2",
            "--dropout",
            "0.2",
            "--patience",
            "5",
            "--seed",
            str(seed),
            "--purge-targets",
            "--num-workers",
            "2",
            "--out-dir",
            str(out_dir),
        ]
        print("\n" + "=" * 80)
        print("RUN", horizon, seed, flush=True)
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        runs.append(
            {
                "horizon": horizon,
                "seed": seed,
                "out_dir": str(out_dir),
            }
        )

manifest = Path("/kaggle/working/purged_run_manifest.json")
manifest.write_text(json.dumps({"purged": True, "runs": runs}, indent=2))
print(f"\nwrote {manifest}")
