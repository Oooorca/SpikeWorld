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
| `spikeworld-verify` | — | Check the immutable paper-number summary |

The repository intentionally excludes exploratory scripts, failed variants,
raw trajectories, checkpoints, caches, and generated result figures. Apart
from the paper overview above, the source tree contains only the final
architecture, adaptation rule, experiment entrypoints, configurations, tests,
and a compact result summary.

## Install

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[train,dev]'
```

Install the optional simulator stack only for the live control experiment:

```bash
pip install -e '.[control]'
```

## Quick checks

These checks need no external artifacts:

```bash
pytest -q
spikeworld-verify
```

They verify the exact model size (`1,451,388` parameters), the exact fast-state
budget (`24,384` bytes), causal sparse-attention accounting, the online input
boundary, and the paper-result summary.

## Artifacts

Large artifacts are kept outside Git. Place them under `artifacts/` according
to [artifacts/README.md](artifacts/README.md). The representative paper
checkpoint must have SHA-256
`6513bc37d9cd3a5a91dedb478ccd54ce87850e6cd06cb1f3fb9437767dece1ff`.
The code loads this checkpoint strictly; all tensor names and shapes must match.

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
[`results/paper_results.json`](results/paper_results.json). This snapshot is not
a substitute for raw trajectories; it is a small guard against transcription
drift between code, tables, and the paper.

## Scope

The released experiments cover registered shear and attenuation families. They
do not establish unrestricted continual learning, discovery of arbitrary
unknown mechanisms, superiority to linear system identification, or measured
neuromorphic-chip energy savings. In the linear attenuation control condition,
RLS has the highest non-oracle mean reward.
