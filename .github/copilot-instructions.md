## Purpose
This repo contains a small Streamlit dashboard and a legacy script that visualize a Gaia sample CSV (`gaia_sample.csv`). The AI agent should focus on making small, safe edits to the dashboard (`milky_way_dashboard.py`) and preserving the simple data load/visualization flow.

## High-level architecture
- Single-process Python visualization project.
- Main interactive app: `milky_way_dashboard.py` (Streamlit).
- Legacy/one-off script: `gaia_old_script.py` (runs a static Plotly chart in browser).
- Data: `gaia_sample.csv` in repo root (required for runs).

## Key patterns and conventions
- Data columns: code expects `l`, `b`, and `parallax` columns. New code should keep these names or map source columns explicitly.
- Distance conversion: use Astropy: `Distance(parallax=... * u.mas, allow_negative=True).pc`. Keep `allow_negative=True` to preserve current handling of negative parallaxes.
- Streamlit caching: `@st.cache_data` is used on `load_data()` in `milky_way_dashboard.py`. When modifying `load_data()`, update or clear the cache appropriately.
- UI updates: `milky_way_dashboard.py` drives UI via Streamlit sidebar controls (e.g., distance slider). Add new controls to sidebar and apply filters to `filtered_df` before plotting.

## How to run locally (developer commands)
- Install runtime dependencies (macOS / Linux):
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install pandas plotly astropy streamlit
  ```
- Run the interactive dashboard:
  ```bash
  streamlit run milky_way_dashboard.py
  ```
- Run the legacy script (non-interactive):
  ```bash
  python gaia_old_script.py
  ```

## Typical code edits the agent may perform
- Add a new sidebar control: add a `st.sidebar.*` widget, apply a mask to `filtered_df`, then regenerate `fig` using `px.scatter`.
- Add columns derived from existing ones: calculate and assign to `df[...]` before caching returns the DataFrame.
- Prefer small, reversible changes; do not rework the UI framework (Streamlit) into a different framework.

## Debugging notes
- Streamlit auto-reloads on file save; use `st.write()` or `st.sidebar.text()` to inspect values.
- For non-Streamlit debugging, `print()` in `gaia_old_script.py` or run with a debugger.

## Integration points & external dependencies
- No external APIs configured. Only external runtime libs are `pandas`, `plotly`, `astropy`, and `streamlit`.
- Data source is the local `gaia_sample.csv`. If fetching remote Gaia files, add explicit download and caching logic in `load_data()` and document it.

## Examples from codebase
- Caching: see `@st.cache_data` above `load_data()` in `milky_way_dashboard.py`.
- Distance conversion: `df["distance_pc"] = Distance(parallax=df["parallax"].values * u.mas, allow_negative=True).pc` in both scripts.
- Plot creation: `px.scatter(..., x="l", y="b", color="distance_pc", ...)`.

## Safety & PR guidance for agents
- Keep changes minimal and focused; prefer separate commits for data-schema changes.
- When changing data column names, update both scripts and include a short migration note in the PR description.

If any of these sections are unclear or you want more detail (testing commands, CI hooks, or a requirements file), tell me which part to expand.
