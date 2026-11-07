# Artifact contract

This contract describes the legacy reference runners and representative checkpoint.
For the current manuscript evidence and full-run availability, see
[`../REPRODUCIBILITY.md`](../REPRODUCIBILITY.md). The current total mutable numeric
budget is 28,848 B including routing history, rather than the legacy tensor-only
24,384 B count.

## Public with this release

- `shifts/`: 20 causal online streams, an online manifest, and a distinct
  post-hoc audit manifest. Mechanism and stage labels occur only in the audit
  manifest.
- `spikeworld_seed42401.pt`: representative legacy checkpoint, included in the prepared code ZIP
  and distributed as a GitHub Release asset through `spikeworld-fetch`.
- `../results/raw/`: immutable joint-training, registered-shift, closed-loop,
  and deployment-audit outputs for the separate legacy result snapshot.
  Current preprint records are in `../results/current/`.

Checksums and the release URL are recorded in `manifest.json`.

## Full experiment layout

The five-seed runners additionally expect the following upstream training
artifacts. These are not silently downloaded or synthesized:

```text
artifacts/
├── multimodal_source.pt
├── spikeworld_seed42401.pt ... spikeworld_seed42405.pt
├── deployment_seed42401.pt ... deployment_seed42405.pt
├── sensory/
│   ├── manifest.json
│   ├── audio_shd.npz
│   ├── audio_ssc.npz
│   ├── retinal_image_text.npz
│   └── video_ssv2.npz
├── action_fit/<task>_fit.npz
├── action_eval/<task>_selection_v2.npz
└── shifts/
```

Each full checkpoint contains `state` and `model_seed`. Each deployment bundle
contains `basis_mean`, `basis`, `router_feature_dim`, and `router_state`.

Online shift shards contain exactly:

```text
current_observations, behavior_actions, observed_endpoint_observations,
episode_seeds, steps, stream_indices
```

They contain no label, teacher output, reward, success flag, mechanism value,
or stage label.

The later three-foundation checkpoints and executors are also not bundled.
Sensory arrays use `train_` and `val_` prefixes: audio/video have `x,y`, retinal
has `image,text,y`, and video also has `teacher_semantic` for the legacy cache
contract. Action arrays use `current_observations`, `candidate_actions`, and
`endpoint_observations`. `src/spikeworld/data.py` defines the exact loading fields.
