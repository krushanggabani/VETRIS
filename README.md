# VETRIS — ViscoElastic Tissue–Robot Interaction Simulation

VETRIS is a lightweight, modular 2D soft-tissue simulator written in Python with a Taichi-backed MLS-MPM core. It focuses on viscoelastic material models, two-way soft–rigid coupling, and tooling for Real2Sim calibration, validation, and parameter optimization.

<p align="center">
  <img src="data/media/base_model.gif" alt="VETRIS demo" width="360"/>
</p>

## Highlights
- MLS-MPM simulation core with P2G/G2P pipeline and stable time-stepping.
- Support for viscoelastic constitutive models (Neo-Hookean + Kelvin–Voigt / SLS-style rate terms).
- Two massager types implemented: `straight` and `dual_arm` (configurable via `CONFIG.engine.massager.type`).
- Two-way soft–rigid contact using proxy bodies and contact-force reporting.
- Real2Sim tooling: signal alignment, composite hysteresis losses, and calibration utilities in `calibrate.py`.
- Small utility modules: plotting (`vetris.io.plotter.Plotter`), structured logging (`data/logs`), and headless rendering.

## Repository layout

- `calibrate.py` — calibration driver and loss utilities used to fit material params to experimental indentation data.
- `optimizer.py` — higher-level optimizer / parameter sets used for automated calibration runs (examples and helpers).
- `run.py` — minimal example that creates a `Simulation(cfg=CONFIG)` and runs it; useful as a quick smoke test.
- `vetris/` — main package (simulation engine, IO, configs, renderers). Key public symbols used in examples:
  - `vetris.config.default_config.CONFIG` — central configuration object.
  - `vetris.engine.sim.Simulation` — top-level simulation runner.
  - `vetris.io.plotter.Plotter` — CSV plotting and simple animations.
- `data/` — contains `real/` experimental data, `calibration/` outputs, `media/` and `gifs/`.
- `logs/` — structured run directories with simulation logs.
- `pyproject.toml`, `VETRIS.egg-info/`, `LICENSE` — packaging and license metadata.


## Quick start

1. Create and activate a virtualenv (Linux / macOS):

```bash
python3 -m venv VRenv
source VRenv/bin/activate
```

2. Install the package in editable mode (this installs the required runtime deps declared in `pyproject.toml`):

```bash
pip install -e .
```

3. Run the minimal example (uses settings from `vetris.config.default_config`):

```bash
python run.py
```

4. Run a calibration trial (example):

```bash
python calibrate.py
```

Notes:
- A Taichi GPU backend is used when available. If you don't have a GPU, Taichi will fall back to CPU; adjust `ti.init` in callers as needed.
- `calibrate.py` expects a CSV at `data/real/force_indentation.csv` (indentation, force). See comments in `calibrate.py` for options and sampling controls.


## Key implemented methods and utilities

- Simulation API
  - Create and run: `sim = Simulation(cfg=CONFIG); sim.run()` or `sim.engine.run()` for stepped control.
  - Access state via `sim.engine.get_state()` which exposes keys like `time`, `deformation.indentation` (in mm) and `contact_force`.

- Calibration helpers (`calibrate.py`)
  - compute_stable_dt(E, nu, rho, n_grid, cfl): CFL-based dt selector for stability.
  - run_sim_and_get_curve(params, coarse=False): run a sim with a parameter vector [E, nu, eta_s, eta_b, rate_k, rate_n] and return (indentation_m, force_N) arrays.
  - loss_composite_hysteresis_safe(...): a robust composite hysteresis-aware loss that aligns simulation to experimental sampling using cumulative |94indentation| progress and computes RMSE, slope/peak/area components.
  - Additional numeric helpers: `_trapz_signed`, `_trapz_abs`, `_robust_initial_slope`, `_robust_slope_segment`, and `_cum_abs_progress` used for alignment and robust metrics.

- Plotting & logging
  - `vetris.io.plotter.Plotter` reads `data/logs/simulation_logs.csv` for quick scatter/line plots and simple animations (MP4/GIF).


## Examples

- Run the engine and write basic logs (from `run.py`):

```bash
python run.py
```

- Quick calibration flow (high-level):
  - Place experimental CSV at `data/real/force_indentation.csv` (two columns: indentation, force).
  - Edit the top of `calibrate.py` to set `MASSAGER_TYPE`, `CSV_IS_MM` or point to a different CSV.
  - Launch `python calibrate.py` to run coarse / fine calibration passes. Outputs land in `data/calibration/exp_1`.


## Dependencies

- Python 3.10+
- taichi
- numpy
- matplotlib
- scipy (optional; `calibrate.py` uses `scipy.optimize.minimize` if available)

Install with `pip install -e .` which uses `pyproject.toml` for dependency metadata.


## License

MIT — see the `LICENSE` file.


## Contact

Open an issue or email: krushgabani95@gmail.com

----
Small edits and additions are welcome; if you'd like I can also add a short example Jupyter notebook or a runnable demo script that records a calibration sweep and saves plots to `data/calibration/`.