# Paper tables rebuilt from raw outputs

| Joint training comparison | Change |
|---|---:|
| Action MSE | 17.10% improvement |
| Next-slice MSE | 0.75% improvement |
| Semantic accuracy | +1.80 pp |
| Binding R@1 | +1.81 pp |

| Registered shift | Prediction MSE | Tracking MSE |
|---|---:|---:|
| Shear | 5.48% improvement | 24.20% improvement |
| Attenuation | 30.01% improvement | 3.94% improvement |

| Closed-loop arm | Mean reward |
|---|---:|
| Frozen | 414.68 |
| SpikeWorld fast state | 422.58 |
| Full tuning | 422.19 |
| RLS | 434.41 |
| Replay | 428.59 |
| Oracle | 485.50 |

Fast-state reward delta: 7.90, 95% CI [2.48, 14.06].

Model parameters: 1,451,388. Mutable fast state: 24,384 bytes. Allowed QK pairs: 58/256 (77.34% reduction).
