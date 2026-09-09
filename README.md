# SpikeWorld

**Adapting Frozen World Models with Compact State**

SpikeWorld retains a pretrained predictive–semantic model and writes deployment
feedback into compact external state. In the paper's registered six-stage stream,
**28.8 KB obtains 19.28% prediction gain**, versus 21.85% for prediction-path tuning
with 1.93 MB of mutable numeric state.

[Paper](paper/SpikeWorld.pdf) · [Architecture](paper/fig1_spikeworld_causal_fast_state.pdf)
· [Memory–prediction comparison](paper/fig2_deployment_pareto.pdf)
· [Reproduction guide](REPRODUCIBILITY.md)

## Predict, observe, then adapt

The frozen sparse spiking model shares temporal computation across sensory and
action streams. Residual matrices improve endpoint predictions; a separate
132-byte score state selects registered action corrections. Outcomes are scored
before updates. Four observed transitions precede the first possible action
correction at decision five.

| Registered-stream method | Prediction gain | Mutable numeric state |
|---|---:|---:|
| External state | 19.28% | 28,848 B |
| Prediction-path tuning | 21.85% | 1,932,488 B |
| Adapter | 15.61% | 135,232 B |
| Replay | 20.38% | 2,693,128 B |

Means use three independently trained foundations, each with three action seeds.
Bytes count persistent mutable numeric state, not the frozen model, transient
activations, or total device/process memory.

## Start here

Use Python **3.11**. Check the current main comparison without ML dependencies:

```bash
python tools/verify_current.py
```

This hash-checks current evidence and reconstructs main-stream prediction,
tracking, return correction, byte counts, and update medians from transition
records and per-run accounting. It does not launch training.

For model and deployment-interface checks:

```bash
uv sync --locked --extra train --extra dev
uv run pytest -q
uv run spikeworld-verify
```

The last command verifies the preserved **legacy** snapshot. Current V5 results
are under `results/current/`; legacy `results/raw/` and `results/paper_results.json`
have not been overwritten.

## Runtime state

`PersistentFastState.accounting()` retains its historical meaning: the 24,384 B
allocated tensor block. `runtime_accounting()` also measures the routing prefix:
four 247-value signatures and four 32-value loss vectors require 4,464 B, giving
**28,848 B peak persistent numeric state**. Prefixes are cleared after routing.
Tests cover all four router outcomes, score accumulation, reset, and causal timing.

## Experiment entry points

| Command | Purpose |
|---|---|
| `spikeworld-joint` | Shared-core multimodal/action optimization |
| `spikeworld-adapt` | Registered-shift prequential evaluation |
| `spikeworld-control` | Live paired Meta-World control |
| `spikeworld-audit` | Model, state, sparsity, and input audit |
| `spikeworld-fetch` | Hash-checked representative legacy checkpoint download |

Read [artifact requirements](artifacts/README.md) and [protocol mapping](REPRODUCIBILITY.md)
before full experiments. These are reference runners, not a turnkey rerun of every
later manuscript experiment.

## Evidence map

`results/current/` contains the main stream, routed actuator RLS, causal closure,
component attribution, closed-loop control, joint formation, formation pilot, and
controlled event-probe records. Every exported JSON retains all numerical fields
and gates; machine-specific paths are anonymized. The manifest records original
and exported hashes.

`results/current/additional_protocol/` retains the later **separate one-foundation**
direct-output NLMS/RLS comparison. It is not pooled with the main stream or plotted
on its frontier. See the reproduction guide for protocol boundaries.

No new experimental results were generated in this release. The linked paper is
the title-only V5 revision of V4. Legacy code and snapshots remain in Git history.
