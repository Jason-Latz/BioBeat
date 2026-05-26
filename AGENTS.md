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
- In Streamlit code, use `width="stretch"` instead of deprecated `use_container_width=True`.
- Cache-only data generation runs should not rewrite existing generated CSVs.
- The BioBeat dashboard should be optimized for guided self data collection: arrows to move between songs, a 30-second rest before each song, and separate HR and EDA charts.
- Do not use `time.sleep()` plus `st.rerun()` loops for Streamlit countdowns; render the countdown client-side and validate elapsed time when the user clicks continue.
- For one-off Python validation commands, set `PYTHONPATH=src` before importing repo packages such as `sensors` or `features`.
- After regenerating tracked CSV artifacts, normalize line endings or verify with `git diff --check`; CRLF output appears as trailing whitespace.
- Documentation should assume Jason runs the app/training pipeline and the sensor teammates only handle hardware/Arduino output; keep guides concise and human-readable.
