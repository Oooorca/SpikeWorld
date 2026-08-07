"""Deployment Audit for parameters, fast state, sparsity, and information access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..adaptation import FAST_BYTES, PersistentFastState
from ..data import ONLINE_ALLOWED_FIELDS
from ..model import local_qk_pairs
from .common import device_from, load_full_checkpoint, read_config, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/audit.toml")
    args = parser.parse_args()
    cfg = read_config(args.config)
    paths, run = cfg["paths"], cfg["run"]
    device = device_from(run.get("device", "cpu"))
    model = load_full_checkpoint(paths["model_checkpoint"], device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    memory = PersistentFastState(torch.zeros(96), torch.eye(96)[:63])
    online = json.loads(Path(paths["online_manifest"]).read_text())
    observed = set()
    for tasks in online["entries"].values():
        for entry in tasks.values():
            observed.update(entry["online_fields"])
    forbidden = observed - ONLINE_ALLOWED_FIELDS
    steps, topk = int(run["sequence_length"]), int(run["attention_topk"])
    sparse_pairs = local_qk_pairs(steps, topk)
    report = {
        "model": {
            "parameter_values": parameter_count,
            "parameter_bytes": parameter_bytes,
        },
        "fast_state": {
            **memory.accounting(),
            "registered_bytes": FAST_BYTES,
            "fraction_of_model": FAST_BYTES / parameter_bytes,
        },
        "attention": {
            "dense_qk_pairs": steps * steps,
            "allowed_sparse_qk_pairs": sparse_pairs,
            "allowed_qk_reduction": 1.0 - sparse_pairs / (steps * steps),
        },
        "online_information": {
            "fields": sorted(observed),
            "forbidden_fields": sorted(forbidden),
        },
    }
    report["qualified"] = (
        parameter_count == 1_451_388
        and memory.accounting()["mutable_bytes"] == FAST_BYTES
        and not forbidden
    )
    write_json(paths["output"], report)


if __name__ == "__main__":
    main()
