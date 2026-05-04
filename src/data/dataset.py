"""
src/data/dataset.py
===================
Shared dataset loader for the PDEBench 2D reaction-diffusion / shallow-water
HDF5 file (``2D_rdb_NA_NA.h5``).

AI Disclaimer
-------------
Coding assistance from ChatGPT and GitHub Copilot was used during development.
The author has thoroughly reviewed, checked, and verified the code for correctness
and takes responsibility for the final implementation used in this project.

The file contains groups keyed by seed string.  Each group has a ``data``
array of shape ``[T, H, W, C]`` where:
    T  = total time steps  (typically 101)
    H  = grid height       (typically 128)
    W  = grid width        (typically 128)
    C  = number of channels (typically 1 for the scalar SWE dataset)

The dataset is split deterministically by seed/trajectory order.
For the reference PDEBench-style procedure, set ``test_ratio=0.1`` and
``val_ratio=0.0``. In that mode the held-out 10% split is reused for both
validation and final evaluation, matching the reference model scripts.

Each sample returned by __getitem__ is a tuple:
    xx : [H, W, initial_step, C]  – autoregressive context window
    yy : [H, W, T, C]             – full trajectory (ground truth)
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _select_indices_near_target(sizes: list[int], target: int) -> list[int]:
    """Return deterministic subset indices with total size closest to target."""
    if target <= 0 or not sizes:
        return []

    # reachable[sum] = (previous_sum, size_index)
    reachable: dict[int, tuple[int, int] | None] = {0: None}
    for idx, size in enumerate(sizes):
        for prev_sum in sorted(list(reachable.keys()), reverse=True):
            new_sum = prev_sum + int(size)
            if new_sum not in reachable:
                reachable[new_sum] = (prev_sum, idx)

    best_sum = min(
        reachable.keys(),
        key=lambda s: (abs(s - target), abs(s), s),
    )

    selected: list[int] = []
    cur = best_sum
    while cur != 0:
        parent = reachable[cur]
        if parent is None:
            break
        prev_sum, idx = parent
        selected.append(idx)
        cur = prev_sum

    selected.sort()
    return selected


@lru_cache(maxsize=16)
def _compute_content_disjoint_split_keys(
    file_path: str,
    test_ratio: float,
    val_ratio: float,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """
    Compute deterministic content-disjoint split keys.

    Keys whose full trajectory arrays are byte-identical are grouped by SHA256,
    then unique-content groups are split into train/val/test so no identical
    trajectory content leaks across splits.
    """
    with h5py.File(file_path, "r") as f:
        all_keys = sorted(f.keys())

        hash_to_keys: dict[str, list[str]] = {}
        for key in all_keys:
            arr = np.array(f[key]["data"], dtype=np.float32)
            digest = hashlib.sha256(arr.tobytes()).hexdigest()
            hash_to_keys.setdefault(digest, []).append(key)

    unique_groups = sorted(hash_to_keys.values(), key=lambda ks: ks[0])
    group_sizes = [len(g) for g in unique_groups]
    total_keys = sum(group_sizes)

    target_test = 0 if test_ratio <= 0 else max(1, int(round(total_keys * test_ratio)))
    target_val = 0 if val_ratio <= 0 else max(1, int(round(total_keys * val_ratio)))

    # Keep at least one trajectory in train if possible.
    if target_test + target_val >= total_keys:
        overflow = (target_test + target_val) - (total_keys - 1)
        if target_val >= overflow:
            target_val -= overflow
        else:
            overflow -= target_val
            target_val = 0
            target_test = max(0, target_test - overflow)

    test_local_idx = _select_indices_near_target(group_sizes, target_test)
    test_idx_set = set(test_local_idx)

    rem_groups = [g for i, g in enumerate(unique_groups) if i not in test_idx_set]
    rem_sizes = [len(g) for g in rem_groups]
    val_local_idx = _select_indices_near_target(rem_sizes, target_val)
    val_idx_set = set(val_local_idx)

    test_groups = [unique_groups[i] for i in sorted(test_idx_set)]
    val_groups = [rem_groups[i] for i in sorted(val_idx_set)]

    val_group_id_set = {id(g) for g in val_groups}
    train_groups = [g for g in rem_groups if id(g) not in val_group_id_set]

    train_keys = tuple(sorted(k for grp in train_groups for k in grp))
    val_keys = tuple(sorted(k for grp in val_groups for k in grp))
    test_keys = tuple(sorted(k for grp in test_groups for k in grp))
    return train_keys, val_keys, test_keys


class SWEDataset(Dataset):
    """
    Torch Dataset wrapping the PDEBench 2D HDF5 file.

    Parameters
    ----------
    filename : str
        HDF5 file stem (e.g. ``"2D_rdb_NA_NA"``).
    saved_folder : str
        Directory that contains ``<filename>.h5``.
    initial_step : int
        Number of time steps used as the autoregressive input context.
    if_test : bool
        If True, return the test split; otherwise return train or val split.
    if_val : bool
        If True (and if_test is False), return the validation split.
    test_ratio : float
        Fraction of seeds reserved for the test split.
    val_ratio : float
        Fraction of seeds reserved for the validation split.
    max_samples : int, optional
        If > 0, cap the number of samples (for quick-mode runs).
    """

    def __init__(
        self,
        filename: str,
        saved_folder: str = "../../data/",
        initial_step: int = 10,
        if_test: bool = False,
        if_val: bool = False,
        test_ratio: float = 0.10,
        val_ratio: float = 0.0,
        max_samples: int = -1,
    ) -> None:
        data_file = f"{filename}.h5"
        workspace_root = Path(__file__).resolve().parents[3]
        project_root = Path(__file__).resolve().parents[2]

        search_dirs: list[Path] = []
        raw_saved = Path(saved_folder)
        if raw_saved.is_absolute():
            search_dirs.append(raw_saved)
        else:
            # Resolve relative to current working directory and project root.
            search_dirs.append(raw_saved.resolve())
            search_dirs.append((project_root / raw_saved).resolve())

        # Hard fallback: always try the workspace data folder.
        search_dirs.append((workspace_root / "data").resolve())

        self.file_path = search_dirs[0] / data_file
        for base_dir in search_dirs:
            candidate = base_dir / data_file
            if candidate.exists():
                self.file_path = candidate
                break

        if not self.file_path.exists():
            tried = "\n  - " + "\n  - ".join(str(d / data_file) for d in search_dirs)
            raise FileNotFoundError(
                "Dataset not found. Tried:" + tried
            )
        self.initial_step = initial_step

        train_keys, val_keys, test_keys = _compute_content_disjoint_split_keys(
            str(self.file_path),
            float(test_ratio),
            float(val_ratio),
        )

        if if_test:
            keys = list(test_keys)
        elif if_val:
            keys = list(val_keys)
        else:
            keys = list(train_keys)

        if max_samples > 0:
            keys = keys[: max_samples]

        self.data_list = np.array(keys)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int):
        """
        Returns
        -------
        xx : FloatTensor [H, W, initial_step, C]
        yy : FloatTensor [H, W, T, C]
        """
        with h5py.File(self.file_path, "r") as f:
            seed_group = f[self.data_list[idx]]
            # HDF5 shape: [T, H, W, C]
            data = np.array(seed_group["data"], dtype=np.float32)

        data = torch.tensor(data)  # [T, H, W, C]

        # Rearrange to [H, W, T, C] — spatial dims first, then time, then channel
        permute_idx = list(range(1, len(data.shape) - 1))  # [1, 2]
        permute_idx.extend([0, len(data.shape) - 1])       # [1, 2, 0, 3]
        data = data.permute(permute_idx)                    # [H, W, T, C]

        xx = data[..., : self.initial_step, :]  # context
        return xx, data


# ---------------------------------------------------------------------------
# FNO-style dataset: includes spatial grid tensor
# ---------------------------------------------------------------------------

class SWEDatasetWithGrid(SWEDataset):
    """
    Extends SWEDataset to also return a normalized spatial grid tensor, as
    expected by FNO2d.  The grid has shape [H, W, 2] with x and y coordinates
    linearly spaced in [0, 1].
    """

    def __getitem__(self, idx: int):
        xx, yy = super().__getitem__(idx)
        H, W = xx.shape[0], xx.shape[1]

        # Build a spatial grid [H, W, 2]
        xs = torch.linspace(0.0, 1.0, H)
        ys = torch.linspace(0.0, 1.0, W)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]

        return xx, yy, grid


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_dataloaders(
    cfg: dict,
    *,
    with_grid: bool = False,
    max_samples: int = -1,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders from a config dict.

    Parameters
    ----------
    cfg : dict
        Configuration dict (typically loaded from configs/default.yaml).
    with_grid : bool
        If True, return SWEDatasetWithGrid loaders (needed for FNO).
    max_samples : int
        Cap on samples per split (quick mode).

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    """
    data_cfg = cfg["data"]
    ds_cls = SWEDatasetWithGrid if with_grid else SWEDataset

    common_kwargs = dict(
        filename=data_cfg["filename"],
        saved_folder=data_cfg["base_path"],
        initial_step=data_cfg["initial_step"],
        test_ratio=data_cfg.get("test_ratio", 0.10),
        val_ratio=data_cfg.get("val_ratio", 0.0),
        max_samples=max_samples,
    )

    train_ds = ds_cls(if_test=False, if_val=False, **common_kwargs)
    if data_cfg.get("val_ratio", 0.0) <= 0:
        # Reference PDEBench-style procedure: the held-out split serves as the
        # validation set during training and as the final evaluation split.
        val_ds = ds_cls(if_test=True, **common_kwargs)
        test_ds = ds_cls(if_test=True, **common_kwargs)
    else:
        val_ds = ds_cls(if_test=False, if_val=True, **common_kwargs)
        test_ds = ds_cls(if_test=True, **common_kwargs)

    tr_cfg = cfg["training"]
    num_workers = data_cfg.get("num_workers", 0)

    train_loader = DataLoader(
        train_ds,
        batch_size=tr_cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=tr_cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=tr_cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader
