"""Verify current manuscript numbers from packaged records; Python stdlib only."""
import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def close(a, b):
    if not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-11):
        raise AssertionError(f"numeric mismatch: {a} != {b}")


def verify(root):
    data = root / "results/current"
    inventory = read(data / "manifest.json")
    for entry in inventory["files"]:
        path = data / entry["file"]
        assert path.stat().st_size == entry["bytes"], entry["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    raw = read(data / "deployment_pareto.json")
    arms = [row["arm"] for row in raw["aggregate"]["pareto_table"]]
    grouped = defaultdict(list)
    assert len(raw["units"]) == 9
    for unit in raw["units"]:
        selected = [r for r in unit["records"] if r["measured"] and r["mechanism"] != "source"]
        returns = [r for r in unit["records"] if r["measured"] and r["stage"] == "source_return"]
        frozen = mean(r["prediction_mse"]["frozen"] for r in selected)
        frozen_tracking = mean(r["tracking_error"]["frozen"] for r in selected)
        computed = {}
        for arm in arms:
            gain = 1 - mean(r["prediction_mse"][arm] for r in selected) / frozen
            tracking = (1 - mean(r["tracking_error"][arm] for r in selected) / frozen_tracking
                        if frozen_tracking > 1e-12 else 0.0)
            correction = mean(r["command_correction_l2"][arm] for r in returns)
            computed[arm] = (gain, tracking, correction)
            for key, value in zip(("prediction_improvement_vs_frozen", "tracking_improvement_vs_frozen",
                                   "source_return_false_correction_l2"), computed[arm]):
                close(value, unit["dynamics"][arm][key])
            if arm == "frozen":
                size = 0
            elif arm == "state_only":
                account = unit["accounting"][arm]
                size = account["allocated_tensor_bytes"] + account["peak_routing_history_numeric_bytes"]
            else:
                account = unit["accounting"][arm]
                size = sum(account[k] for k in ("mutable_parameter_bytes", "optimizer_tensor_bytes",
                           "score_state_allocated_tensor_bytes", "score_state_peak_routing_history_numeric_bytes",
                           "replay_buffer_bytes"))
            close(size, next(r["mutable_numeric_bytes"] for r in raw["aggregate"]["pareto_table"] if r["arm"] == arm))
        grouped[unit["foundation_seed"]].append(computed)
    assert len(grouped) == 3 and all(len(g) == 3 for g in grouped.values())
    table = []
    for row in raw["aggregate"]["pareto_table"]:
        arm = row["arm"]
        for index, key in enumerate(("mean_prediction_gain", "mean_tracking_gain", "mean_source_return_false_correction_l2")):
            rebuilt = mean(mean(u[arm][index] for u in units) for units in grouped.values())
            close(rebuilt, row[key])
        latency = None if arm == "frozen" else median(
            value for u in raw["units"] for value in u["update_latency"][arm]["samples_ms"])
        table.append({**row, "median_update_ms": latency})
    close(next(r["mean_prediction_gain"] for r in table if r["arm"] == "state_only"), 0.1928373138914193)
    control = read(data / "closed_loop_control.json")
    legacy = read(root / "results/raw/closed_loop_control.json")
    for arm, row in control["aggregate"]["arms"].items():
        close(row["mean_reward"], legacy["aggregate"]["arms"][arm]["mean_reward"])
    return {"status": "PASS", "hash_checked_files": len(inventory["files"]),
            "independent_foundations": len(grouped), "paired_units": len(raw["units"]),
            "pareto_reconstructed_from_transition_records": table,
            "other_evidence": "hash-checked; not all statistics recomputed by this command"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))
