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

## Current manuscript mapping

| Paper evidence | File in `results/current/` | Unit / setting |
|---|---|---|
| Figure 3A,C / Table 2 | `deployment_pareto.json` | 3 foundations × 3 action seeds |
| Figure 3B | `routed_matrix_rls.json` | same-stream actuator identification |
| Table 1 / causal timing | `causal_closure.json` | bounded/legacy trajectory checks |
| Table 3 | `component_ablation.json` | full versus score-only |
| Table 4 | `closed_loop_control.json` | 5 models × 3 tasks × 5 replicas per arm |
| Section 6.4 | `joint_training.json` | 5 action initializations, one sensory foundation |
| Table 5 | `formation_bridge.json` | one-foundation development pilot |
| Figure 2C | `formation_probes.json`, `probe_records/` | separate controlled event models |
| Source-return provenance | `source_reuse.json` | calibrated-source reentry |

## Protocol details

The stream uses registered shear and attenuation values. Shifted archives are
separate from calibration; source returns reuse calibration episodes and their
four-observation prefixes. Zero return correction measures calibrated reentry,
not unseen-source generalization. The online API does not read evaluator labels,
reward, or success. Matrices improve prediction; score state and the registered
inverse produce action changes. Prediction-path tuning is not whole-model tuning.

Historical event probes use separate encoders and predictive plus contrastive
objectives. Identity metadata constructs paired views although labels are absent
from the prediction loss. The composition probe supplies factor segments and
supports the image/audio factor-association comparison, not text-side composition
or arbitrary multimodal grounding.

Closed-loop recorded intervals use 20,000 paired bootstrap draws, independently
resampling model, task, and environment axes. The title-only manuscript's phrase
"model-seed clustering" is abbreviated and should be read with this implementation
detail. Main-stream intervals instead use foundation-level clustering.

## Additional, separate protocol

The one-foundation fresh-trajectory comparison in `additional_protocol/` includes
a summary and four paired unit records per method plus frozen references. It
reports hidden-state gain 23.34%, direct-output NLMS 51.70%, and direct-output RLS
45.84%, with 28,848 / 13,956 / 21,156 B respectively. These are not independent
foundation replications of Table 2. The routed actuator RLS comparison does not
exhaust classical output calibration. All these method arms are retained.

## Export integrity and environment

`results/current/manifest.json` maps exported files to original SHA-256 values.
Only path strings and JSON formatting change; numeric fields, booleans, nulls,
comparison arms, and gates are retained. Source hashes inside archived metadata
refer to original workspace artifacts, not necessarily packaged copies. Use the
export manifest to verify packaged bytes. Internal reviews are not included.

The locked dependencies are in `uv.lock`. Use Python 3.11 (`tomllib` is required).
GPU timing is historical CUDA-synchronized timing; CPU smoke tests do not reproduce
those latency measurements. Tests and current-result verification need no simulator.
