# Clean-Reverse Hardware Checks

Created: June 1, 2026 at 11:20:50 CDT

This checkpoint preserves the one-song `clip_001` Raspberry Pi Seeed GSR/EDA checks captured before clean reverse block `01`.

## Captures

| Session | Time | Rows | Audit status | Median adjacent EDA change | Fraction of changes >= 0.02 |
| --- | --- | ---: | --- | ---: | ---: |
| `self_20260601_004314` | Overnight check | `2451` | `questionable_noise_or_transient` | `0.0096` | `0.115` |
| `self_20260601_111921` | Latest check | `2439` | `questionable_noise_or_transient` | `0.0048` | `0.034` |

The latest check is materially better than the overnight check and does not resemble the disconnected-lead artifact. A CSV cannot prove physical sensor placement, so Jason must still confirm corrected placement and firm lead attachment before block `01`.
