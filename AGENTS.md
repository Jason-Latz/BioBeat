# Global AGENTS.md

Machine-level rules for all agent work in this repository.

## Mandatory Learning Capture

Any time an agent learns either:

1. A mistake the agent made, or
2. A user preference, especially Jason's preferences,

the agent must update this file to record that learning as an explicit, actionable rule.

Do not leave that learning only in chat history.

## Learned Rules

- Prefer many small, narrowly scoped commits for BioBeat implementation and GitHub publishing work.
- In Streamlit code, prefer `width="stretch"` when the installed Streamlit version supports it; otherwise use a compatibility helper or `use_container_width=True` because Streamlit 1.37.1 raises `TypeError` for `width`.
- Cache-only data generation runs should not rewrite existing generated CSVs.
- The BioBeat dashboard should be optimized for guided self data collection: arrows to move between songs, a 30-second rest before each song, and separate HR and EDA charts.
- Do not use `time.sleep()` plus `st.rerun()` loops for Streamlit countdowns; render the countdown client-side and validate elapsed time when the user clicks continue.
- For one-off Python validation commands, set `PYTHONPATH=src` before importing repo packages such as `sensors` or `features`.
- After regenerating tracked CSV artifacts, normalize line endings or verify with `git diff --check`; CRLF output appears as trailing whitespace.
- Documentation should assume Jason runs the app/training pipeline, the sensor teammates handle Seeed GSR/EDA output from a Raspberry Pi, and Apple Watch HR is imported from CSV after collection; keep guides concise and human-readable.
- Rating UI should explain that valence is the numeric positive/negative model label and mood is only an optional human-readable tag.
- BioBeat collection UI should use one explicit control to start song playback and sensor capture together; avoid separate manual play/record steps when possible.
- Split BioBeat commits by purpose even for small follow-ups: separate source data edits, generated artifact updates, docs, and tooling/instruction changes.
- BioBeat rating flow should auto-save ratings from the forward navigation action; avoid a separate save button in the normal self-training path.
- When adding BioBeat songs for training, favor genre diversity and keep batches small enough that self-collection sessions remain practical.
- BioBeat recording code and docs should refer to Seeed GSR/EDA serial input from a Raspberry Pi, not Arduino, and should allow HR and EDA to arrive as separate raw sensor rows.
- BioBeat dashboard should instruct users to start an Apple Watch Workout before collection and provide a clear end-of-session Apple Watch HR CSV upload/import step with per-song sync coverage.
- BioBeat serial setup should prioritize real USB serial devices such as `/dev/cu.usbmodem*` and avoid defaulting to Mac Bluetooth/debug console ports.
- Seeed GSR serial lines may send `milliseconds_since_start,eda`; store the first value as `sensor_elapsed_ms` and never display or treat it as heart rate.
- BioBeat's next modeling direction should derive arousal from EDA plus Apple Watch PPG/heart-rate response, collect valence as negative/neutral/positive self-report, train both valence classification and discrete regression experiments, and support a short 5-song new-user calibration pass.
- BioBeat calibration should use the short new-user sensor pass to estimate baseline biometric behavior, adjust normalization, and optionally calibrate valence probabilities rather than fully retraining complex models from only 5 samples.
- Do not build continuous real-time mood guessing for BioBeat yet; focus on initial calibration of the generic model for each user.
- Current BioBeat collected training data is outdated and not recorded well; remove stale label/sensor/derived training artifacts when resetting the dataset, but keep the folder structure for future collection.
