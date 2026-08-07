"""Registered-Shift Adaptation experiment."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..adaptation import MechanismRouter, PersistentFastState, candidate_spec
from ..data import load_online_shard
from ..model import TASKS, tensor_hash
from .common import device_from, load_full_checkpoint, read_config, write_json


def _posthoc(mechanism: str, command: np.ndarray) -> np.ndarray:
    output = command.copy()
    if mechanism == "shear":
        output[1] = np.clip(command[1] + 0.45 * command[0], -1.0, 1.0)
    elif mechanism == "attenuation":
        output[:2] = 0.40 * command[:2]
    return output


def process(
    model,
    router,
    memory,
    shard: dict[str, np.ndarray],
    task_index: int,
    device: torch.device,
    evidence_actions: np.ndarray | None = None,
) -> list[dict]:
    previous, records = None, []
    for index in range(len(shard["steps"])):
        episode = int(shard["episode_seeds"][index])
        if episode != previous:
            memory.start_episode()
            previous = episode
        state = torch.from_numpy(shard["current_observations"][index : index + 1]).to(
            device
        )
        action = torch.from_numpy(shard["behavior_actions"][index : index + 1]).to(device)
        evidence = action
        if evidence_actions is not None:
            evidence = torch.from_numpy(evidence_actions[index : index + 1]).to(device)
        endpoint = torch.from_numpy(
            shard["observed_endpoint_observations"][index : index + 1]
        ).to(device)
        task = torch.tensor([task_index], dtype=torch.long, device=device)
        started = time.perf_counter()
        with torch.no_grad():
            target = model.target_standardized(state, endpoint)
            frozen, _ = model.predict_standardized(state, action, task)
            adapted = memory.predict(model, state, action, task)
            command = memory.inverse(action)[0].cpu().numpy()
            route_before = memory.route
        report = memory.observe(
            model, router, state, action, task, endpoint, evidence_action=evidence
        )
        records.append(
            {
                "episode_seed": episode,
                "stream_index": int(shard["stream_indices"][index]),
                "route": route_before,
                "wrote": report["wrote"],
                "frozen_prediction_loss": float(F.mse_loss(frozen, target)),
                "adapted_prediction_loss": float(F.mse_loss(adapted, target)),
                "desired_action": shard["behavior_actions"][index].tolist(),
                "adapted_command": command.tolist(),
                "latency_seconds": time.perf_counter() - started,
            }
        )
    return records


def annotate(records: list[dict], stage_map: list[dict]) -> None:
    audit = {int(row["seed"]): row for row in stage_map}
    seen: dict[int, int] = {}
    for row in records:
        episode = row["episode_seed"]
        local = seen.get(episode, 0)
        seen[episode] = local + 1
        mechanism = audit[episode]["mechanism"]
        desired = np.asarray(row.pop("desired_action"), dtype=np.float32)
        command = np.asarray(row.pop("adapted_command"), dtype=np.float32)
        behavior = _posthoc(mechanism, desired)
        executed = _posthoc(mechanism, command)
        row.update(
            {
                "stage": audit[episode]["stage"],
                "mechanism": mechanism,
                "measured": local >= 4,
                "behavior_tracking_error": float(
                    np.sum((behavior[:2] - desired[:2]) ** 2)
                ),
                "adapted_tracking_error": float(
                    np.sum((executed[:2] - desired[:2]) ** 2)
                ),
            }
        )


def shuffled_episode_actions(shard: dict[str, np.ndarray], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = shard["behavior_actions"].copy()
    for episode in np.unique(shard["episode_seeds"]):
        rows = np.flatnonzero(shard["episode_seeds"] == episode)
        output[rows] = output[rng.permutation(rows)]
    return output


def shard_path(entry: dict, manifest_path: str | Path) -> Path:
    recorded = Path(entry["path"])
    if recorded.is_file():
        return recorded
    relocated = Path(manifest_path).parent / recorded.name
    if not relocated.is_file():
        raise FileNotFoundError(relocated)
    return relocated


def summarize(records: list[dict]) -> list[dict]:
    output = []
    for stage in sorted({row["stage"] for row in records}):
        local = [row for row in records if row["stage"] == stage and row["measured"]]
        frozen = np.mean([row["frozen_prediction_loss"] for row in local])
        adapted = np.mean([row["adapted_prediction_loss"] for row in local])
        behavior = np.mean([row["behavior_tracking_error"] for row in local])
        tracking = np.mean([row["adapted_tracking_error"] for row in local])
        mechanism = local[0]["mechanism"]
        output.append(
            {
                "stage": stage,
                "mechanism": mechanism,
                "prediction_improvement": 1.0 - adapted / frozen,
                "action_improvement": 0.0
                if behavior <= 1e-12
                else 1.0 - tracking / behavior,
                "route_accuracy": float(
                    np.mean([row["route"] == mechanism for row in local])
                ),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/adaptation.toml")
    args = parser.parse_args()
    cfg = read_config(args.config)
    paths, run = cfg["paths"], cfg["run"]
    device = device_from(run.get("device", "cpu"))
    families, values = candidate_spec(device)
    online = json.loads(Path(paths["online_manifest"]).read_text())
    pending, integrity = [], {}
    checkpoints = paths["model_checkpoints"]
    bundles = paths["deployment_bundles"]
    for model_index, (checkpoint, bundle_path) in enumerate(
        zip(checkpoints, bundles, strict=True)
    ):
        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        model = load_full_checkpoint(checkpoint, device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        frozen_hash = tensor_hash(model)
        router = MechanismRouter(int(bundle["router_feature_dim"])).to(device)
        router.load_state_dict(bundle["router_state"], strict=True)
        router.freeze()
        for replica_number, (replica_name, tasks) in enumerate(online["entries"].items()):
            for arm in ("treatment", "action_shuffled"):
                memory = PersistentFastState(
                    bundle["basis_mean"].to(device),
                    bundle["basis"].to(device),
                    families,
                    values,
                ).to(device)
                for task_index, task in enumerate(TASKS):
                    shard = load_online_shard(
                        shard_path(tasks[task], paths["online_manifest"])
                    )
                    evidence = None
                    if arm == "action_shuffled":
                        evidence = shuffled_episode_actions(
                            shard, model_index * 10_000 + replica_number * 1_000 + task_index
                        )
                    records = process(
                        model,
                        router,
                        memory,
                        shard,
                        task_index,
                        device,
                        evidence,
                    )
                    pending.append(
                        {
                            "model_index": model_index,
                            "replica": replica_name,
                            "task": task,
                            "arm": arm,
                            "records": records,
                        }
                    )
        integrity[str(model_index)] = {
            "model_parameters_bitwise_unchanged": tensor_hash(model) == frozen_hash
        }

    # The audit manifest is deliberately opened only after all online arms finish.
    audit = json.loads(Path(paths["audit_manifest"]).read_text())
    cells = []
    for item in pending:
        annotate(
            item["records"],
            audit["entries"][item["replica"]][item["task"]]["stage_audit_map"],
        )
        cells.extend(
            {
                **row,
                "model_index": item["model_index"],
                "replica": item["replica"],
                "task": item["task"],
                "arm": item["arm"],
            }
            for row in summarize(item["records"])
        )
    aggregate = {}
    for mechanism in ("shear", "attenuation"):
        local = [
            row
            for row in cells
            if row["mechanism"] == mechanism and row["arm"] == "treatment"
        ]
        shuffled = {
            (row["model_index"], row["replica"], row["task"], row["stage"]): row
            for row in cells
            if row["mechanism"] == mechanism and row["arm"] == "action_shuffled"
        }
        advantages = [
            row["action_improvement"]
            - shuffled[
                (row["model_index"], row["replica"], row["task"], row["stage"])
            ]["action_improvement"]
            for row in local
        ]
        aggregate[mechanism] = {
            "prediction_improvement_mean": float(
                np.mean([row["prediction_improvement"] for row in local])
            ),
            "action_improvement_mean": float(
                np.mean([row["action_improvement"] for row in local])
            ),
            "route_accuracy": float(np.mean([row["route_accuracy"] for row in local])),
            "action_improvement_difference_from_shuffled": float(np.mean(advantages)),
        }
    write_json(
        paths["output"],
        {
            "experiment": "registered_shift_adaptation",
            "aggregate": aggregate,
            "cells": cells,
            "integrity": {
                "models": integrity,
                "audit_manifest_opened_after_online_arms": True,
                "fast_state_bytes": memory.accounting()["mutable_bytes"],
                "online_fields": sorted(
                    next(iter(online["entries"].values()))[TASKS[0]]["online_fields"]
                ),
            },
        },
    )


if __name__ == "__main__":
    main()
