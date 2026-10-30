#!/usr/bin/env python
"""Build the per-well manifest used by the benchmark documentation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qc", default="data/processed/qc_full.csv")
    parser.add_argument("--full", default="data/processed/panel_full.parquet")
    parser.add_argument("--core", default="data/processed/panel_core.parquet")
    parser.add_argument("--clean", default="data/processed/panel_core_clean.parquet")
    parser.add_argument("--out", default="docs/site_manifest.csv")
    return parser.parse_args()


def sites(path: Path) -> set[str]:
    frame = pd.read_parquet(PROJECT_ROOT / path, columns=["site_no"])
    return set(frame["site_no"].astype(str).unique())


def main() -> None:
    args = parse_args()
    qc = pd.read_csv(PROJECT_ROOT / args.qc, dtype={"site_no": str})
    meta = pd.read_parquet(
        PROJECT_ROOT / args.full, columns=["site_no", "state_cd"]
    ).drop_duplicates("site_no")
    meta["site_no"] = meta["site_no"].astype(str)
    full = sites(Path(args.full))
    core = sites(Path(args.core))
    clean = sites(Path(args.clean))

    out = qc.merge(meta, on="site_no", how="left")
    out["in_full"] = out["site_no"].isin(full)
    out["in_core"] = out["site_no"].isin(core)
    out["in_clean"] = out["site_no"].isin(clean)

    def reason(row: pd.Series) -> str:
        if not row["in_full"]:
            return "not in selected full panel"
        if not row["in_core"]:
            return "coverage < 70%"
        if not row["in_clean"]:
            return ">5 jumps > 5 ft"
        return ""

    out["exclusion_reason"] = out.apply(reason, axis=1)
    columns = [
        "site_no", "state_cd", "n_days", "n_observed", "coverage", "n_jumps",
        "max_abs_jump", "n_gaps_gt_7d", "longest_gap_days", "value_min",
        "value_max", "value_std", "in_full", "in_core", "in_clean",
        "exclusion_reason",
    ]
    out = out[columns]
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(out)} wells)")
    print(out["exclusion_reason"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
