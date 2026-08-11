# Artifact contract

## Public with this release

- `shifts/`: 20 causal online streams, an online manifest, and a distinct
  post-hoc audit manifest. Mechanism and stage labels occur only in the audit
  manifest.
- `spikeworld_seed42401.pt`: representative paper checkpoint, distributed as a
  GitHub Release asset and installed by `spikeworld-fetch`.
- `../results/raw/`: immutable joint-training, registered-shift, closed-loop,
  and deployment-audit outputs used to rebuild all reported summary numbers.

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
