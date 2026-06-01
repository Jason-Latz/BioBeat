# Clean-Reverse Block 01 Checkpoint

Created: June 1, 2026 after the completed `self_20260601_112658` run.

This checkpoint preserves the first completed clean-reverse block:

```text
clip_240
clip_239
...
clip_201
```

## Audit Summary

| Metric | Value |
| --- | ---: |
| Saved labels | `40` |
| Sensor trials | `40` |
| Sensor rows | `97840` |
| Missing EDA values | `0` |
| Missing elapsed-time values | `0` |
| Non-positive elapsed-time steps | `0` |
| Approximate sample rate | `47.62 Hz` |
| Median adjacent EDA change range | `0.0033` through `0.0048` |
| Adjacent-change fraction >= `0.02` range | `0.025` through `0.062` |

The conservative serial-pattern audit labels the trials `questionable_noise_or_transient`, but the block is consistent across all `40` songs and does not resemble the known disconnected-lead artifact. Jason confirmed corrected sensor placement and firm lead attachment before collection. This is the cleanest real BioBeat EDA block collected so far and is approved as the first primary biometric-training candidate.

The per-trial values are preserved in `eda_qc_manifest.csv`.
