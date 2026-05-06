"""
src/utils/logger.py
====================
Experiment logging: CSV training curves + JSON final summary.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path


class ExperimentLogger:
    """
    Logs training curves to a CSV file and final metrics to a JSON file.

    Parameters
    ----------
    model_name : str
        Identifier string, e.g. ``"unet"``.
    log_dir : str | Path
        Directory where ``<model_name>_log.csv`` will be written.
    results_dir : str | Path
        Directory where ``<model_name>_metrics.json`` will be written.
    """

    def __init__(
        self,
        model_name: str,
        log_dir: str = "results/logs",
        results_dir: str = "results",
        append: bool = False,
    ) -> None:
        self.model_name = model_name
        self.log_dir = Path(log_dir)
        self.results_dir = Path(results_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self._csv_path = self.log_dir / f"{model_name}_log.csv"
        append_mode = append and self._csv_path.exists()
        self._csv_file = self._csv_path.open("a" if append_mode else "w", newline="")
        self._writer = csv.writer(self._csv_file)
        if (not append_mode) or self._csv_path.stat().st_size == 0:
            self._writer.writerow(["epoch", "train_loss", "val_loss", "elapsed_s"])
            self._csv_file.flush()

        self._start = time.time()
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
    ) -> None:
        elapsed = time.time() - self._start
        self._writer.writerow([epoch, f"{train_loss:.6e}", f"{val_loss:.6e}", f"{elapsed:.1f}"])
        self._csv_file.flush()
        self.history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        )

    # ------------------------------------------------------------------
    def save_metrics(self, metrics: dict) -> None:
        """
        Save a final metrics dict to ``<results_dir>/<model_name>_metrics.json``.
        """
        out = {
            "model": self.model_name,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **metrics,
        }
        path = self.results_dir / f"{self.model_name}_metrics.json"
        with path.open("w") as f:
            json.dump(out, f, indent=2)
        print(f"[Logger] Metrics saved → {path}")

    # ------------------------------------------------------------------
    def close(self) -> None:
        self._csv_file.close()

    def __del__(self) -> None:
        try:
            self._csv_file.close()
        except Exception:
            pass
