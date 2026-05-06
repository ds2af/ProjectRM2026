"""
scripts/copy_runtime_significance.py
====================================
Initialize runtime significance directory (standalone).

This script creates the runtime_significance directory structure for
storing repeated training experiments and statistical analysis results.
It's self-contained and doesn't depend on external backups.

Usage
-----
    python scripts/copy_runtime_significance.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DST = ROOT / "results" / "runtime_significance"


def main():
    DST.mkdir(parents=True, exist_ok=True)
    
    # Create placeholder files if they don't exist
    files = [
        ("runtime_significance_summary.json", {"status": "not_run"}),
        ("repeated_runs.csv", "repeat,seed,model,rmse,rel_l2,max_error,inference_time_s,n_params\n"),
    ]
    
    created = []
    for fname, content in files:
        fpath = DST / fname
        if not fpath.exists():
            if isinstance(content, dict):
                with fpath.open("w") as f:
                    json.dump(content, f, indent=2)
            else:
                with fpath.open("w") as f:
                    f.write(content)
            created.append(fname)
            print(f"[created] {fname}")
        else:
            print(f"[exists] {fname}")
    
    print(f"\n[runtime_significance] Ready at {DST}")
    if created:
        print(f"[runtime_significance] Created {len(created)} placeholder file(s)")


if __name__ == "__main__":
    main()
