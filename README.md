# SpikeWorld

Reference code for the experiments in **SpikeWorld: Fast-State Adaptation for
Frozen Spiking World Models**.

SpikeWorld uses one sparse spiking Transformer to learn next-slice prediction,
multimodal semantics, image--text binding, and action-conditioned dynamics. At
deployment, the trained network is frozen. Only a 24,384-byte external state is
updated from the observed next-state prediction error.

![SpikeWorld overview](images/spikeworld-hero-causal-fast-state-v3.png)

## What is included

| Command | Experiment | Purpose |
|---|---|---|
| `spikeworld-joint` | Joint Multimodal–Action Training | Five-seed shared-core optimization |
| `spikeworld-adapt` | Registered-Shift Adaptation | Prequential shear/attenuation adaptation |
| `spikeworld-control` | Closed-Loop Control | Live Meta-World comparison |
| `spikeworld-audit` | Deployment Audit | Parameter, state-size, sparsity, and input audit |
| `spikeworld-verify` | Result provenance | Rebuild and check every compact paper number |
| `spikeworld-fetch` | Artifact retrieval | Download and hash-check the public checkpoint |

The source tree contains the final architecture, adaptation rule, experiment
entrypoints, configurations, tests, causal shift streams, and the four raw JSON
outputs from which the paper tables are rebuilt. Exploratory variants and
caches are intentionally excluded.

## Install

Python 3.10 or newer and [uv](https://docs.astral.sh/uv/) are required for the
locked environment.

```bash
uv sync --locked --extra train --extra dev
```

Install the optional simulator stack only for the live control experiment:

```bash
uv sync --locked --extra control
```

## Quick checks

These checks need no external artifacts:

```bash
uv run pytest -q
uv run spikeworld-verify
```

They verify the exact model size (`1,451,388` parameters), the exact fast-state
budget (`24,384` bytes), causal sparse-attention accounting, the online input
boundary, same-input semantic-logit invariance, decision-5 causal timing, and
raw-output reconstruction of the paper summary.

## Artifacts

The registered-shift streams and their separately held audit manifest are
versioned under `artifacts/shifts/`. Download the representative paper
checkpoint from the immutable GitHub release with:

```bash
uv run spikeworld-fetch
```

The downloader verifies SHA-256
`6513bc37d9cd3a5a91dedb478ccd54ce87850e6cd06cb1f3fb9437767dece1ff`
before installing the file. The complete artifact boundary is documented in
[artifacts/README.md](artifacts/README.md).

## Reproduce

Run commands from the repository root. Paths and hyperparameters are ordinary
TOML files and can be changed without editing source code.

```bash
# Joint training (a two-update pipeline check is available with --smoke)
spikeworld-joint --config configs/joint.toml --smoke
spikeworld-joint --config configs/joint.toml

# Offline prequential evaluation; audit labels are opened only after updates
spikeworld-adapt --config configs/adaptation.toml

# Live paired control; --smoke uses one model, task, and environment seed
spikeworld-control --config configs/control.toml --smoke
spikeworld-control --config configs/control.toml

# Static deployment audit
spikeworld-audit --config configs/audit.toml
```

The online update API accepts only `state`, `action`, `task`, and `endpoint`.
It rejects labels, teacher outputs, rewards, success flags, and true mechanism
parameters. Reward and success are read by the control runner only after the
transition and are used only for reporting.

## Paper-result snapshot

The locked experiments report:

| Result | Value |
|---|---:|
| Joint-training action MSE improvement | 17.10% |
| Joint-training semantic accuracy change | +1.80 pp |
| Shear prediction / action improvement | 5.48% / 24.20% |
| Attenuation prediction / action improvement | 30.01% / 3.94% |
| Frozen to fast-state reward | 414.68 to 422.58 |
| Frozen to fast-state success | 53.33% to 66.67% |
| Sparse QK reduction | 77.34% |

The full-precision values are in
[`results/paper_results.json`](results/paper_results.json). Unlike a hard-coded
snapshot check, `spikeworld-verify` derives this file from the archived outputs
in `results/raw/`. To rewrite the JSON and the human-readable tables:

```bash
uv run spikeworld-verify --write
```

## Scope

The released experiments cover registered shear and attenuation families. They
do not establish unrestricted continual learning, discovery of arbitrary
unknown mechanisms, superiority to linear system identification, or measured
neuromorphic-chip energy savings. In the linear attenuation control condition,
RLS has the highest non-oracle mean reward. The public representative checkpoint
and causal shift streams support integrity and interface checks; a full
five-seed retraining run still requires the upstream sensory and action-fitting
datasets listed in the artifact contract.
