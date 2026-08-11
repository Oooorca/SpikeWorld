"""Rebuild and verify the paper summary from immutable raw experiment outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RAW_FILES = {
    "joint": "joint_training.json",
    "adaptation": "registered_shift_adaptation.json",
    "control": "closed_loop_control.json",
    "system": "deployment_audit.json",
}


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def build(raw_dir: str | Path) -> dict:
    """Derive every compact paper number from the archived raw JSON outputs."""
    root = Path(raw_dir)
    joint = _read(root / RAW_FILES["joint"])
    adaptation = _read(root / RAW_FILES["adaptation"])
    control = _read(root / RAW_FILES["control"])
    audit = _read(root / RAW_FILES["system"])

    joint_aggregate = joint["aggregate"]
    adaptation_aggregate = adaptation["aggregate"]
    arms = control["aggregate"]["arms"]
    fast_effect = control["aggregate"]["effects_vs_frozen"]["safe_fast"]
    sparse = audit["efficiency"]["sparse_attention_proxy"]
    treatment_episodes = {
        (row["replica"], row["task"], row["episode_seed"])
        for row in adaptation["records"]
        if row["arm"] == "treatment"
    }

    return {
        "adaptation": {
            "attenuation_action_improvement": adaptation_aggregate["attenuation"][
                "action_improvement_mean"
            ],
            "attenuation_prediction_improvement": adaptation_aggregate["attenuation"][
                "prediction_improvement_mean"
            ],
            "attenuation_route_accuracy": adaptation_aggregate["attenuation"][
                "route_accuracy"
            ],
            "episodes": len(treatment_episodes),
            "shear_action_improvement": adaptation_aggregate["shear"][
                "action_improvement_mean"
            ],
            "shear_prediction_improvement": adaptation_aggregate["shear"][
                "prediction_improvement_mean"
            ],
            "shear_route_accuracy": adaptation_aggregate["shear"]["route_accuracy"],
        },
        "control": {
            "fast_reward": arms["safe_fast"]["mean_reward"],
            "fast_reward_delta": fast_effect["reward_delta"]["mean"],
            "fast_reward_delta_ci95": fast_effect["reward_delta"]["ci95"],
            "fast_success": arms["safe_fast"]["success_rate"],
            "frozen_reward": arms["frozen"]["mean_reward"],
            "frozen_success": arms["frozen"]["success_rate"],
            "full_tuning_reward": arms["full_tuning"]["mean_reward"],
            "oracle_reward": arms["oracle"]["mean_reward"],
            "replay_reward": arms["replay"]["mean_reward"],
            "rls_reward": arms["rls"]["mean_reward"],
        },
        "joint": {
            "action_mse_improvement": joint_aggregate["action_mse_improvement"]["mean"],
            "binding_r1_change": joint_aggregate["binding_r1_change"]["mean"],
            "next_slice_improvement": joint_aggregate["next_slice_improvement"]["mean"],
            "semantic_accuracy_change": joint_aggregate["semantic_accuracy_change"][
                "mean"
            ],
            "seeds": joint["model_seeds"],
        },
        "system": {
            "allowed_qk_reduction": sparse["allowed_qk_reduction"],
            "allowed_sparse_qk_pairs": sparse["allowed_sparse_qk_pairs"],
            "dense_qk_pairs": sparse["dense_qk_pairs"],
            "fast_state_bytes": audit["efficiency"]["fast_state_bytes"],
            "model_parameter_bytes_fp32": audit["model"]["total_parameter_bytes_fp32"],
            "model_parameters": audit["model"]["total_parameter_values"],
            "safe_fast_update_ms": audit["efficiency"]["safe_fast_update_ms"],
            "software_forward_speedup": sparse["measured_full_forward_speedup"],
        },
    }


def _compare(expected, observed, path: str = "") -> dict:
    if isinstance(expected, dict) and isinstance(observed, dict):
        mismatches = {}
        for key in sorted(expected.keys() | observed.keys()):
            location = f"{path}.{key}" if path else key
            if key not in expected or key not in observed:
                mismatches[location] = {
                    "expected": expected.get(key),
                    "observed": observed.get(key),
                }
            else:
                mismatches.update(_compare(expected[key], observed[key], location))
        return mismatches
    if isinstance(expected, list) and isinstance(observed, list):
        if len(expected) != len(observed):
            return {path: {"expected": expected, "observed": observed}}
        mismatches = {}
        for index, (left, right) in enumerate(zip(expected, observed, strict=True)):
            mismatches.update(_compare(left, right, f"{path}[{index}]"))
        return mismatches
    if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
        if abs(float(expected) - float(observed)) <= 1e-12:
            return {}
    elif expected == observed:
        return {}
    return {path: {"expected": expected, "observed": observed}}


def verify(path: str | Path, raw_dir: str | Path | None = None) -> None:
    snapshot_path = Path(path)
    raw_root = snapshot_path.parent / "raw" if raw_dir is None else Path(raw_dir)
    snapshot = _read(snapshot_path)
    rebuilt = build(raw_root)
    mismatches = _compare(snapshot, rebuilt)
    if mismatches:
        raise AssertionError(json.dumps(mismatches, indent=2, sort_keys=True))


def write_markdown(data: dict, path: str | Path) -> None:
    joint, adapt, control, system = (
        data["joint"],
        data["adaptation"],
        data["control"],
        data["system"],
    )
    text = f"""# Paper tables rebuilt from raw outputs

| Joint training comparison | Change |
|---|---:|
| Action MSE | {100 * joint['action_mse_improvement']:.2f}% improvement |
| Next-slice MSE | {100 * joint['next_slice_improvement']:.2f}% improvement |
| Semantic accuracy | +{100 * joint['semantic_accuracy_change']:.2f} pp |
| Binding R@1 | +{100 * joint['binding_r1_change']:.2f} pp |

| Registered shift | Prediction MSE | Tracking MSE |
|---|---:|---:|
| Shear | {100 * adapt['shear_prediction_improvement']:.2f}% improvement | {100 * adapt['shear_action_improvement']:.2f}% improvement |
| Attenuation | {100 * adapt['attenuation_prediction_improvement']:.2f}% improvement | {100 * adapt['attenuation_action_improvement']:.2f}% improvement |

| Closed-loop arm | Mean reward |
|---|---:|
| Frozen | {control['frozen_reward']:.2f} |
| SpikeWorld fast state | {control['fast_reward']:.2f} |
| Full tuning | {control['full_tuning_reward']:.2f} |
| RLS | {control['rls_reward']:.2f} |
| Replay | {control['replay_reward']:.2f} |
| Oracle | {control['oracle_reward']:.2f} |

Fast-state reward delta: {control['fast_reward_delta']:.2f}, 95% CI [{control['fast_reward_delta_ci95'][0]:.2f}, {control['fast_reward_delta_ci95'][1]:.2f}].

Model parameters: {system['model_parameters']:,}. Mutable fast state: {system['fast_state_bytes']:,} bytes. Allowed QK pairs: {system['allowed_sparse_qk_pairs']}/{system['dense_qk_pairs']} ({100 * system['allowed_qk_reduction']:.2f}% reduction).
"""
    Path(path).write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="results/paper_results.json")
    parser.add_argument("--raw-dir", default="results/raw")
    parser.add_argument("--write", action="store_true", help="rewrite JSON and Markdown")
    parser.add_argument("--tables", default="results/paper_tables.md")
    args = parser.parse_args()
    if args.write:
        data = build(args.raw_dir)
        Path(args.path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        write_markdown(data, args.tables)
        print(f"rebuilt: {args.path}, {args.tables}")
    else:
        verify(args.path, args.raw_dir)
        print(f"verified against raw outputs: {args.path}")


if __name__ == "__main__":
    main()
