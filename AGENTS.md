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
- BioBeat self-training should be streamlined and fast for long Jason-run sessions: auto-start each 20-second rest period, auto-start song playback plus sensor recording after rest, use native Streamlit rating buttons for emotion and familiarity, use emotion `1=negative, 2=neutral, 3=positive`, use familiarity `1-5`, and auto-advance after familiarity.
- Do not rely on custom JavaScript, iframe components, hidden global keyboard listeners, or autofocus hacks for BioBeat Streamlit ratings; Jason saw digits highlight the Begin/restart button and the blue capture box failed, so use native Streamlit controls for reliability.
- When Jason asks to add songs for BioBeat training, include broad genre coverage across jazz, classical, rap, EDM, pop, 80s, metal, Latin, and other styles when the requested batch size allows it.
- Use `.venv/bin/python` for BioBeat project scripts that depend on `requirements.txt`; the system `python3` may not have packages such as `requests` installed.
- If the iTunes API returns 403 for a combined song and artist search term, retry narrower/simplified query terms before treating the row as failed.
- Avoid repeated rapid iTunes batch/preflight runs; Apple can return 429 rate limits, so validate CSV structure offline first and rerun preview generation later or with throttling.
- For iTunes clip generation, cache successful candidate lookups and resume only missing desired songs so repeated runs do not re-query already resolved tracks.
- After regenerating `data/clips.csv`, restart the Streamlit experiment app or reset the session because `st.cache_data` and `st.session_state.order` can keep showing the old clip count.
- For real BioBeat collection sessions, launch `src/dashboard/app.py`; `src/experiment/runner_streamlit.py` is a ratings-only collector and must not be used for Raspberry Pi GSR/EDA capture.
- During live Raspberry Pi GSR/EDA collection, save every serial row but throttle Streamlit chart redraws; per-sample chart rendering can make the reader fall behind the sensor stream.
- Keep the BioBeat live GSR/EDA monitor far below the primary collection controls behind a large vertical scroll gap so Jason can inspect it only when needed without seeing it during normal collection.
- Do not add a visibility widget for the BioBeat live sensor monitor. Jason prefers a fixed below-the-fold monitor because Streamlit widget reruns disrupted active timers.
- Flush BioBeat raw sensor rows to disk incrementally during each timed phase and once more when leaving the phase; do not wait until the full rest/listen phase ends to persist every buffered row.
- The Raspberry Pi Seeed GSR stream can emit one bare numeric startup line whenever the serial reader opens. Require `sensor_elapsed_ms` for dashboard Seeed samples so startup values such as `74` or `36` are not stored as EDA outliers.
- Jason is frustrated by repeated unverified BioBeat collector UI changes. Before making another collection-flow change, audit every Streamlit rerun path and verify the exact active-recording behavior end to end without disrupting Jason's browser tab.
- Do not start a temporary Streamlit validation server or navigate the active in-app browser away from Jason's real collector during BioBeat one-take setup. Use isolated verification that cannot replace or redirect the user's current tab.
- Do not add or interact with Streamlit widgets during an active rest/listen capture until rerun behavior is proven safe. Streamlit widget interactions can rerun the script and reset the song timer.
- Do not use browser automation to test Jason's active BioBeat collector tab during one-take setup. Validate from source and isolated commands, then ask Jason to perform the short manual UI dry run.
- During BioBeat setup, consistently direct Jason to the intended collector URL `http://127.0.0.1:8503/`; do not refer him back to temporary validation ports such as `8504`.
- Never rewrite `data/clips.csv` or change the active clip registry during a BioBeat collection take. Finish or deliberately end the current session first, then regenerate clips and begin a separate session.
- Use Jason's completed familiarity ratings to build future supplemental BioBeat batches around unfamiliar or deeper-cut tracks. Do not respond to an over-familiar batch by adding more broadly popular songs.
- For BioBeat deeper-cut iTunes continuation batches, use a conservative request delay of roughly 6.5 seconds, persist each successful lookup immediately, and resume from cache rather than making rapid repeated API requests.
- When replacing an over-familiar BioBeat queue, retain all completed labeled trials as real training data and replace only the uncollected remainder in a separate continuation session.
- Before a long real BioBeat collection take, run a short hardware-fit dry run and audit EDA adjacent-sample changes. A disconnected GSR lead can produce a highly repetitive oscillation rather than a flat line, so visible variance alone does not prove that the signal is valid.
- Treat BioBeat ratings and biometric validity separately. Preserve completed ratings when hardware quality fails, but exclude questionable GSR/EDA blocks from primary biometric training until they are recollected or explicitly modeled as noisy secondary data.
- Use `negative`, `neutral`, and `positive` for BioBeat human emotion labels. Treat legacy `sad` and `happy` values as backwards-compatible aliases only.

## Active Issues Before Next Attempt

- The in-app browser was redirected from the intended collector on port `8503` to a temporary validation instance on port `8504`. Restore one known collector URL before further testing.
- Confirm manually that the fixed below-the-fold sensor monitor stays out of view during normal collection while raw GSR rows continue saving.
- The `test89` Raspberry Pi dry run after the startup-line parser guard passed: both rest and listen phases saved at about 49 Hz with no missing `sensor_elapsed_ms`, malformed EDA values, timing reversals, or startup outliers.
- Jason manually confirmed that the `test89` collector flow passed visually: the below-the-fold EDA monitor placement, uninterrupted rest/song timers, and audio playback all behaved correctly.
- Before Jason starts a real BioBeat collection take, archive prior `self_*` label and sensor CSVs into a timestamped `data/archive/` folder instead of deleting them. Leave tracked demo fixtures and placeholders in the active raw folders.
- Keep BioBeat session setup locked during timed capture, but provide `Start a new session` after capture reaches ratings or completion so Jason can transition from a dry run to the real take without restarting Streamlit.
- Do not tell Jason the collector is ready again until his non-disruptive dry run proves timer continuity, audio continuity, incremental raw sensor persistence, and below-the-fold monitor placement together.
