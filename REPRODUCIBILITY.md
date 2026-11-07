# Reproduction and evidence guide

## Three levels of use

1. **Recorded-result verification (CPU, standard library):** run `python tools/verify_current.py`.
   The main table is recomputed from nine per-run transition records, averaging action
   seeds within foundation before averaging foundations. Byte counts come from each
   arm's ledger; latency medians use recorded samples. Other evidence files are
   hash-checked, not fully re-analyzed by this command.
2. **Executable interface checks:** install the package, run `pytest -q`, then
   `spikeworld-fetch` and `spikeworld-audit --config configs/audit.toml`. The
   representative checkpoint is legacy seed 42401, not one of the three independent
   foundations underlying the current Pareto experiment.
3. **Fresh experiments:** the reference joint/adapt/control runners require the
   training caches, fitted router/PCA bundles, and checkpoints in `artifacts/README.md`.
   This distribution does not include a standalone runner and every input needed
   to rerun all current three-foundation experiments or historical formation probes
   from scratch. Stored-output verification is not a fresh experimental reproduction.

## Current preprint mapping

| Paper evidence | File in `results/current/` | Unit / setting |
|---|---|---|
| Prediction–storage trade-off / Figure 3 | `deployment_pareto.json` | 3 foundations × 3 action seeds |
| Actuator identification | `routed_matrix_rls.json` | separate matched-stream replay |
| Behavioral equivalence | `causal_closure.json` | 36 traces in 9 four-controller scenarios |
| Component roles | `component_ablation.json` | full versus score-only |
| Closed-loop reward / Figure 4 | `closed_loop_control.json` | 5 models × 3 tasks × 5 replicas per arm |
| Joint training and three-stage appendix table | `joint_training.json` | 5 action initializations, one sensory foundation |
| Training-stage appendix pilot | `formation_bridge.json` | one-foundation development pilot |
| Controlled probes | `formation_probes.json`, `probe_records/` | separate event models |
| Direct output calibration appendix | `additional_protocol/` | one foundation, 4 paired units per arm |
| Source-return provenance | `source_reuse.json` | calibrated-source reentry |

## Protocol details

The stream uses registered shear and attenuation values. Shifted archives are
separate from calibration; source returns reuse calibration episodes and their
four-observation prefixes. Zero return correction measures calibrated reentry,
not unseen-source generalization. The online API does not read evaluator labels,
reward, or success. Matrices improve prediction; score state and the registered
inverse produce action changes. Prediction-path tuning is not whole-model tuning.
Every main run replays 348 common transitions, including 174 shifted transitions.
Reported gains use 126 shifted transitions after each episode's four-transition
routing prefix. Tracking applies the known actuator map to corrected commands;
live environment reward is measured in the separate control study.

Historical event probes use separate encoders and predictive plus contrastive
objectives. Identity metadata constructs paired views although labels are absent
from the prediction loss. The composition probe supplies factor segments and
supports the image/audio factor-association comparison, not text-side composition
or arbitrary multimodal grounding.

Closed-loop recorded intervals use 20,000 paired bootstrap draws, independently
resampling model, task, and replica axes while preserving method pairing.
Main-stream intervals instead use foundation-level clustering.
The legacy reference runner `src/spikeworld/experiments/control.py` uses a flat
episode bootstrap for its own summary. Its confidence intervals are not the
three-axis intervals reported in the paper; the archived current evidence
contains the paper's statistics.
Every closed-loop episode has attenuation 0.4 and 13 decisions / 104 environment
steps. Adaptation carries across replicas within a model/task pair. The tuning
arm uses AdamW on non-semantic parameters; gain-memory replay retrieves local
gain estimates with nearest neighbors. These differ from the main stream's
Adam prediction-path tuning and neural replay. The oracle has no 0.05 correction
cap and clips only to the valid command range.

The archived joint-learning sensory metric averages four groups (audio combines
SHD and SSC). The reference evaluator averages five streams and uses validation
persistence denominators. Training uses training-set denominators. The preprint
preserves the archived metric definition.

## Inputs for fresh reference runs

The included seed-42401 checkpoint and shift streams support the executable
interface checks above. Fresh `spikeworld-joint`, `spikeworld-adapt`, and
`spikeworld-control` runs additionally require these unbundled inputs:

- `artifacts/spikeworld_seed42402.pt` through `spikeworld_seed42405.pt`;
- `artifacts/deployment_seed42401.pt` through `deployment_seed42405.pt`;
- `artifacts/multimodal_source.pt`;
- `artifacts/sensory/`, `artifacts/action_fit/`, and `artifacts/action_eval/`.

The current three-foundation experiment also requires its original checkpoints
and executor. Those are not supplied by the legacy runners. This release supports
verification of recorded results and executable interface tests; it does not
provide a complete build recipe for these missing inputs.

## Additional, separate protocol

The one-foundation fresh-trajectory comparison in `additional_protocol/` includes
a summary and four paired unit records per method plus frozen references. It
reports hidden-state gain 23.34%, direct-output NLMS 51.70%, and direct-output RLS
45.84%, with 28,848 / 13,956 / 21,156 B respectively. These are not independent
foundation replications of the main-stream table. The routed actuator RLS comparison does not
exhaust classical output calibration. All these method arms are retained.

## Export integrity and environment

`results/current/manifest.json` maps exported files to original SHA-256 values.
Path strings are anonymized and internal workflow decision labels are replaced
by neutral experimental-status labels. Numeric fields, booleans, nulls,
comparison arms, and gates are retained. Source hashes inside archived metadata
refer to original workspace artifacts, not necessarily packaged copies. Use the
export manifest to verify packaged bytes. Internal reviews are not included.

The locked dependencies are in `uv.lock`. Use Python 3.11 (`tomllib` is required).
GPU timing is historical CUDA-synchronized timing; CPU smoke tests do not reproduce
those latency measurements. Tests and current-result verification need no simulator.
