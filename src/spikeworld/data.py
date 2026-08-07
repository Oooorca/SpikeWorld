"""Load the compact array contracts used by the paper experiments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .model import ACTION_DIM, STATE_DIM, TASKS

SENSORY_FILES = {
    "audio_shd": "audio_shd.npz",
    "audio_ssc": "audio_ssc.npz",
    "retinal": "retinal_image_text.npz",
    "video": "video_ssv2.npz",
}
ONLINE_ALLOWED_FIELDS = {
    "current_observations",
    "behavior_actions",
    "observed_endpoint_observations",
    "episode_seeds",
    "steps",
    "stream_indices",
}


def _tensor(values: np.ndarray, dtype: torch.dtype | None = None) -> torch.Tensor:
    output = torch.from_numpy(np.asarray(values))
    return output.to(dtype=dtype) if dtype is not None else output


def load_sensory(root: str | Path) -> tuple[dict[str, object], dict[str, object]]:
    root = Path(root)
    missing = [
        name
        for name in (*SENSORY_FILES.values(), "manifest.json")
        if not (root / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing sensory artifacts: {missing}")
    archives = {name: np.load(root / file) for name, file in SENSORY_FILES.items()}
    arrays: dict[str, object] = {
        "audio_shd": {
            split: (
                _tensor(archives["audio_shd"][f"{split}_x"], torch.float32),
                _tensor(archives["audio_shd"][f"{split}_y"], torch.long),
            )
            for split in ("train", "val")
        },
        "audio_ssc": {
            split: (
                _tensor(archives["audio_ssc"][f"{split}_x"], torch.float32),
                _tensor(archives["audio_ssc"][f"{split}_y"], torch.long),
            )
            for split in ("train", "val")
        },
        "retinal": {
            split: (
                _tensor(archives["retinal"][f"{split}_image"], torch.float32),
                _tensor(archives["retinal"][f"{split}_text"], torch.float32),
                _tensor(archives["retinal"][f"{split}_y"], torch.long),
            )
            for split in ("train", "val")
        },
        "video": {
            split: (
                _tensor(archives["video"][f"{split}_x"], torch.float32),
                _tensor(archives["video"][f"{split}_y"], torch.long),
                _tensor(archives["video"][f"{split}_teacher_semantic"], torch.float32),
            )
            for split in ("train", "val")
        },
    }
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    return arrays, manifest


def sensory_loaders(
    arrays: dict[str, object], batch_size: int, seed: int
) -> dict[str, DataLoader]:
    output = {}
    for offset, source in enumerate(("audio_shd", "audio_ssc", "retinal", "video")):
        for split in ("train", "val"):
            generator = torch.Generator().manual_seed(seed + 2 * offset + (split == "val"))
            output[f"{split}_{source}"] = DataLoader(
                TensorDataset(*arrays[source][split]),
                batch_size=batch_size,
                shuffle=split == "train",
                drop_last=split == "train",
                generator=generator,
                num_workers=0,
            )
    return output


def persistence_denominators(manifest: dict[str, object], split: str) -> dict[str, float]:
    return {
        "audio_shd": float(manifest["audio_shd"][f"{split}_persistence_mse"]),
        "audio_ssc": float(manifest["audio_ssc"][f"{split}_persistence_mse"]),
        "image": float(manifest["retinal_image_text"][f"{split}_image_persistence_mse"]),
        "text": float(manifest["retinal_image_text"][f"{split}_text_persistence_mse"]),
        "video": float(manifest["video"][f"{split}_persistence_mse"]),
    }


def load_action_arrays(root: str | Path, suffix: str) -> dict[str, np.ndarray]:
    root = Path(root)
    states, actions, endpoints, tasks = [], [], [], []
    for task_index, task in enumerate(TASKS):
        path = root / f"{task}_{suffix}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as archive:
            current = archive["current_observations"].astype(np.float32)
            candidate = archive["candidate_actions"].astype(np.float32)
            endpoint = archive["endpoint_observations"].astype(np.float32)
        count, candidates = candidate.shape[:2]
        states.append(
            np.repeat(current[:, None], candidates, axis=1).reshape(-1, STATE_DIM)
        )
        actions.append(candidate.reshape(-1, ACTION_DIM))
        endpoints.append(endpoint.reshape(-1, STATE_DIM))
        tasks.append(np.full(count * candidates, task_index, dtype=np.int64))
    return {
        "state": np.concatenate(states),
        "action": np.concatenate(actions),
        "endpoint": np.concatenate(endpoints),
        "task": np.concatenate(tasks),
    }


def delta_statistics(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    delta = data["endpoint"] - data["state"]
    return delta.mean(0).astype(np.float32), np.maximum(delta.std(0), 1e-4).astype(
        np.float32
    )


def action_batch(
    data: dict[str, np.ndarray], indices: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.from_numpy(data[name][indices]).to(device)
        for name in ("state", "action", "task", "endpoint")
    )


def load_online_shard(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        fields = set(archive.files)
        unexpected = fields - ONLINE_ALLOWED_FIELDS
        missing = ONLINE_ALLOWED_FIELDS - fields
        if unexpected or missing:
            raise ValueError(
                f"online contract mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        return {name: archive[name] for name in archive.files}
