"""Verify the compact, immutable paper-result summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPECTED = {
    "joint.action_mse_improvement": 0.1709561687693975,
    "joint.next_slice_improvement": 0.007535254543733538,
    "joint.semantic_accuracy_change": 0.01795238095238095,
    "joint.binding_r1_change": 0.018125001713633536,
    "adaptation.shear_prediction_improvement": 0.05479861529864915,
    "adaptation.shear_action_improvement": 0.2419739605392585,
    "adaptation.attenuation_prediction_improvement": 0.30005303976853337,
    "adaptation.attenuation_action_improvement": 0.039416927626359635,
    "control.frozen_reward": 414.68016416788913,
    "control.fast_reward": 422.58465320554353,
    "control.fast_reward_delta": 7.904489037654288,
    "control.fast_success": 0.6666666666666666,
    "system.model_parameters": 1_451_388,
    "system.fast_state_bytes": 24_384,
}


def _lookup(data: dict, dotted: str):
    value = data
    for key in dotted.split("."):
        value = value[key]
    return value


def verify(path: str | Path) -> None:
    data = json.loads(Path(path).read_text())
    mismatches = {}
    for name, expected in EXPECTED.items():
        observed = _lookup(data, name)
        if abs(float(observed) - expected) > 1e-12:
            mismatches[name] = {"expected": expected, "observed": observed}
    if mismatches:
        raise AssertionError(json.dumps(mismatches, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="results/paper_results.json")
    args = parser.parse_args()
    verify(args.path)
    print(f"verified: {args.path}")


if __name__ == "__main__":
    main()
