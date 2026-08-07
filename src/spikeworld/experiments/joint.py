"""Joint Multimodal--Action Training experiment."""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import torch

from ..data import delta_statistics, load_action_arrays, load_sensory
from ..model import ModelConfig, SemanticWorldModel, SpikeWorld
from ..training import evaluate_model, train_control_path, train_joint
from .common import bootstrap, device_from, read_config, write_json


def source_model(
    checkpoint: Path,
    action_fit: dict,
    seed: int,
    device: torch.device,
    control_updates: int,
) -> SpikeWorld:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source_cfg = payload.get("config", {})
    config = ModelConfig(
        sequence_length=int(source_cfg.get("sequence_length", 16)),
        width=int(source_cfg.get("embed_dim", 96)),
        layers=int(source_cfg.get("layers", 2)),
        heads=int(source_cfg.get("heads", 4)),
        ff_dim=int(source_cfg.get("ff_dim", 192)),
        dropout=float(source_cfg.get("dropout", 0.0)),
        beta=float(source_cfg.get("beta", 0.85)),
        attention_topk=int(source_cfg.get("attention_topk", 4)),
    )
    anchored = SemanticWorldModel(config)
    anchored.load_state_dict(payload["state"], strict=True)
    mean, scale = delta_statistics(action_fit)
    model = SpikeWorld(config, mean, scale)
    model.anchored.load_state_dict(anchored.state_dict(), strict=True)
    model.to(device)
    train_control_path(model, action_fit, seed=seed, updates=control_updates, device=device)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/joint.toml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = read_config(args.config)
    paths, run = cfg["paths"], cfg["run"]
    device = device_from(run.get("device", "auto"))
    sensory, manifest = load_sensory(paths["sensory_root"])
    action_fit = load_action_arrays(paths["action_fit_root"], "fit")
    action_eval = load_action_arrays(paths["action_eval_root"], "selection_v2")
    seeds = run["seeds"][:1] if args.smoke else run["seeds"]
    updates = 2 if args.smoke else int(run["joint_updates"])
    control_updates = 2 if args.smoke else int(run["control_updates"])
    started = time.time()
    rows, representative = [], None
    for seed in seeds:
        base = source_model(
            Path(paths["source_checkpoint"]), action_fit, seed, device, control_updates
        )
        baseline = evaluate_model(base, sensory, manifest, action_eval, device)
        arm_results = {}
        for arm in ("action_only", "joint"):
            model = copy.deepcopy(base)
            curve = train_joint(
                model,
                sensory,
                manifest,
                action_fit,
                seed=seed,
                updates=updates,
                batch_size=int(run["batch_size"]),
                learning_rate=float(run["learning_rate"]),
                arm=arm,
                device=device,
            )
            arm_results[arm] = {
                "metrics": evaluate_model(model, sensory, manifest, action_eval, device),
                "curve": curve,
            }
            if representative is None and arm == "joint":
                representative = {"model_seed": seed, "state": model.state_dict()}
        joint = arm_results["joint"]["metrics"]
        action_only = arm_results["action_only"]["metrics"]
        effects = {
            "action_mse_improvement": 1.0 - joint["action_mse"] / baseline["action_mse"],
            "next_slice_improvement": 1.0
            - joint["mean_normalized_next_slice"] / baseline["mean_normalized_next_slice"],
            "semantic_accuracy_change": joint["mean_semantic_accuracy"]
            - baseline["mean_semantic_accuracy"],
            "binding_r1_change": joint["binding_mean_class_r1"]
            - baseline["binding_mean_class_r1"],
            "joint_vs_action_only_action_mse_relative": joint["action_mse"]
            / action_only["action_mse"]
            - 1.0,
            "joint_vs_action_only_binding_r1": joint["binding_mean_class_r1"]
            - action_only["binding_mean_class_r1"],
        }
        rows.append(
            {"seed": seed, "baseline": baseline, "arms": arm_results, "effects": effects}
        )
    aggregate = {
        name: bootstrap([row["effects"][name] for row in rows], 20_000, 43310 + i)
        for i, name in enumerate(rows[0]["effects"])
    }
    output = Path(paths["output"])
    write_json(
        output,
        {
            "experiment": "joint_multimodal_action",
            "config": cfg,
            "aggregate": aggregate,
            "rows": rows,
            "runtime_seconds": time.time() - started,
        },
    )
    torch.save(representative, output.with_suffix(".pt"))


if __name__ == "__main__":
    main()
