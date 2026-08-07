# Artifact layout

Artifacts are not committed to Git. The experiment configurations expect:

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
├── action_fit/
│   └── <task>_fit.npz
├── action_eval/
│   └── <task>_selection_v2.npz
└── shifts/
    ├── online_manifest.json
    ├── audit_manifest.json
    └── replica<0-4>_<task>_two_shift.npz
```

Each full model checkpoint contains `state` and `model_seed`. Each deployment
bundle contains `basis_mean`, `basis`, `router_feature_dim`, and `router_state`.

Online shift shards contain exactly:

```text
current_observations, behavior_actions, observed_endpoint_observations,
episode_seeds, steps, stream_indices
```

Mechanism and stage labels belong only in `audit_manifest.json`; they must not
appear in an online shard.
