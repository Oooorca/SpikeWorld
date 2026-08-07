from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import tomllib
import torch

from ..model import ModelConfig, SpikeWorld


def read_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def write_json(path: str | Path, value: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def device_from(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_full_checkpoint(path: str | Path, device: torch.device) -> SpikeWorld:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["state"]
    if "delta_mean" not in state or "delta_scale" not in state:
        raise ValueError("expected a full SpikeWorld checkpoint with action statistics")
    model = SpikeWorld(ModelConfig(), state["delta_mean"], state["delta_scale"])
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


def bootstrap(values: list[float], draws: int, seed: int) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = array[rng.integers(0, len(array), size=(draws, len(array)))].mean(1)
    return {
        "mean": float(array.mean()),
        "ci95": np.quantile(samples, (0.025, 0.975)).tolist(),
        "values": array.tolist(),
        "draws": draws,
        "seed": seed,
    }
