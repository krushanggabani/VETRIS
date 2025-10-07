from typing import List, Tuple, Optional, Dict, Literal
import os, csv, time
import numpy as np

# Reuse your optional matplotlib loader
def _ensure_mpl():
    import importlib
    return importlib.import_module("matplotlib.pyplot")

ScaleMode = Literal["auto", "linear", "log"]

def _safe_eval(engine, objective, x, exp_i, exp_f, *, coarse=True):
    """Runs engine + objective with robust failure handling."""
    try:
        si, sf, meta = engine.run(x, coarse=coarse)
        f, comps, npts = objective.evaluate(x, exp_i, exp_f, si, sf, meta=meta)
        return float(f), comps, npts, si, sf, meta
    except Exception as e:
        si = np.asarray([], float); sf = np.asarray([], float)
        meta = {"finished": False, "progress": 0.0, "reason": f"exception:{type(e).__name__}"}
        return float(1e12), {}, 0, si, sf, meta

def _geom_or_linspace(bounds: Tuple[float,float], num: int, mode: ScaleMode) -> np.ndarray:
    lo, hi = float(bounds[0]), float(bounds[1])
    if mode == "linear":
        return np.linspace(lo, hi, num)
    if mode == "log":
        lo_pos = max(lo, np.finfo(float).tiny)
        return np.geomspace(lo_pos, hi, num)
    # auto: choose log if both positive and cover >= 1 decade
    if lo > 0 and hi/lo >= 10.0:
        return np.geomspace(lo, hi, num)
    return np.linspace(lo, hi, num)

class ParamSensitivity:
    """
    One-at-a-time (OAT) parameter sensitivity sweeps.
    - Varies one parameter over a grid (linear/log/auto)
    - Holds others fixed at x_ref
    - Logs via plotter (log_trial, log_curve, plot_curves, log_progress/plot_progress)
    - Saves detailed CSV per-parameter + an overall summary CSV
    """
    def __init__(
        self,
        engine_runner,
        objective,
        plotter,
        bounds: List[tuple],
        x_ref: List[float],
        param_names: List[str],
        out_dir: Optional[str] = None,
        seed: int = 1234,
    ):
        self.engine = engine_runner
        self.obj = objective
        self.plot = plotter
        self.bounds = list(bounds)
        self.x_ref = np.asarray(x_ref, float).copy()
        self.param_names = list(param_names)
        self.rng = np.random.default_rng(seed)

        # Output dirs/files
        self.base_dir = out_dir or getattr(self.plot, "final_dir", os.getcwd())
        os.makedirs(self.base_dir, exist_ok=True)
        self.summary_csv = os.path.join(self.base_dir, "sensitivity_summary.csv")
        if not os.path.exists(self.summary_csv):
            with open(self.summary_csv, "w", newline="") as f:
                csv.writer(f).writerow([
                    "param", "best_value", "best_objective", "n_evals", "timestamp"
                ])

        # internal running trial id (for consistent file naming with your system)
        self._trial_counter = 0

    def _detail_csv_path(self, pname: str) -> str:
        safe = pname.replace("/", "_")
        return os.path.join(self.base_dir, f"sens_{safe}.csv")

    def _init_detail_csv(self, pname: str):
        path = self._detail_csv_path(pname)
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "param", "value", "objective",
                    "finished", "progress", "reason",
                    "E", "nu", "eta_s", "eta_b", 
                    "rate_k", "rate_n",
                    # common comps (if present they’ll be filled, else blank):
                    "rmse", "rmse_load", "rmse_unload",
                    "n_load", "n_unload",
                    "d_peak", "d_slope", "d_area",
                    "slope_exp_head", "slope_exp_tail",
                    "slope_sim_head", "slope_sim_tail",
                    "area_exp", "area_sim"
                ])

    def _append_detail_row(self, pname: str, value: float, fval: float, x: np.ndarray, meta: dict, comps: Dict[str, float]):
        path = self._detail_csv_path(pname)
        def fmt(v):
            try: return f"{float(v):.6e}"
            except: return ""
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                pname, fmt(value), fmt(fval),
                meta.get("finished", None), meta.get("progress", None), meta.get("reason", None),
                # params (assumes 6 as per your system; adjust if you add more):
                *[fmt(v) for v in x.tolist()],
                # comps in a stable order (blank if missing)
                fmt(comps.get("rmse", "")),
                fmt(comps.get("rmse_load", "")),
                fmt(comps.get("rmse_unload", "")),
                comps.get("n_load", ""),
                comps.get("n_unload", ""),
                fmt(comps.get("d_peak", "")),
                fmt(comps.get("d_slope", "")),
                fmt(comps.get("d_area", "")),
                fmt(comps.get("slope_exp_head", "")),
                fmt(comps.get("slope_exp_tail", "")),
                fmt(comps.get("slope_sim_head", "")),
                fmt(comps.get("slope_sim_tail", "")),
                fmt(comps.get("area_exp", "")),
                fmt(comps.get("area_sim", "")),
            ])

    def _append_summary_row(self, pname: str, best_val: float, best_f: float, n_evals: int):
        with open(self.summary_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                pname, f"{best_val:.6e}", f"{best_f:.6e}", n_evals, time.strftime("%Y-%m-%d %H:%M:%S")
            ])

    def analyze_param(
        self,
        exp_i: np.ndarray,
        exp_f: np.ndarray,
        param_index: int,
        *,
        grid_points: int = 15,
        scale: ScaleMode = "auto",
        value_override: Optional[np.ndarray] = None,
        coarse: bool = True,
        do_plot: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Sweep one parameter while holding others at x_ref.
        Returns (values, objectives, best_value, best_objective).
        """
        pname = self.param_names[param_index]
        self._init_detail_csv(pname)

        vals = value_override if value_override is not None else \
               _geom_or_linspace(self.bounds[param_index], grid_points, scale)
        vals = np.asarray(vals, float)

        fvals = np.full(vals.shape, np.nan, float)
        best_f, best_v = np.inf, float(vals[0])

        for i, v in enumerate(vals):
            x = self.x_ref.copy()
            x[param_index] = float(np.clip(v, self.bounds[param_index][0], self.bounds[param_index][1]))

            # global trial id (aligns with your logging/plot files)
            self._trial_counter += 1
            tid = self._trial_counter

            f, comps, npts, si, sf, meta = _safe_eval(self.engine, self.obj, x, exp_i, exp_f, coarse=coarse)
            fvals[i] = f
            if f < best_f:
                best_f, best_v = f, x[param_index]

            # Your plotter hooks (same calls as BayesianTPE)
            if hasattr(self.plot, "log_progress"): self.plot.log_progress(f)
            if hasattr(self.plot, "plot_progress"): self.plot.plot_progress()
            if hasattr(self.plot, "plot_curves"):
                self.plot.plot_curves(
                    exp_i, exp_f, si, sf,
                    title=f"Sensitivity [{pname}={x[param_index]:.3e}]  f={f:.3e}",
                    out_name=f"sens_{pname}_{tid:03d}.png",
                    to_runs=True
                )
            if hasattr(self.plot, "log_trial"):
                # save the scalar + comps to your main log CSV
                self.plot.log_trial(tid, x, f, comps, meta)
            if hasattr(self.plot, "log_curve"):
                self.plot.log_curve(tid, si, sf)

            # also append to the per-parameter CSV
            self._append_detail_row(pname, x[param_index], f, x, meta, comps)

        # Per-parameter summary
        self._append_summary_row(pname, best_v, best_f, int(vals.size))

        # Optional quick plot (objective vs value)
        if do_plot:
            try:
                plt = _ensure_mpl()
                fig, ax = plt.subplots(1, 1, figsize=(5.6, 4.0), constrained_layout=True)
                if (scale == "log") or (scale == "auto" and self.bounds[param_index][0] > 0 and self.bounds[param_index][1]/self.bounds[param_index][0] >= 10.0):
                    ax.semilogx(vals, fvals, marker="o")
                else:
                    ax.plot(vals, fvals, marker="o")
                ax.set_title(f"Sensitivity: {pname}")
                ax.set_xlabel(pname)
                ax.set_ylabel("Objective")
                ax.grid(True, alpha=0.3)
                # mark best
                ax.scatter([best_v], [best_f], s=50, marker="*", zorder=5)
                fig.savefig(os.path.join(self.base_dir, f"sens_{pname}_summary.png"), dpi=140)
                plt.close(fig)
            except Exception:
                pass

        return vals, fvals, float(best_v), float(best_f)

    def analyze_all(
        self,
        exp_i: np.ndarray,
        exp_f: np.ndarray,
        *,
        grid_points: int = 15,
        scale_per_param: Optional[List[ScaleMode]] = None,
        coarse: bool = True,
        do_plot: bool = True,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Run sensitivity for every parameter.
        Returns a dict:
          { pname: {"values": vals, "objectives": fvals, "best_value": v*, "best_objective": f*}, ... }
        """
        res: Dict[str, Dict[str, np.ndarray]] = {}
        if scale_per_param is None:
            scale_per_param = ["auto"] * len(self.param_names)
        for j, pname in enumerate(self.param_names):
            vals, fvals, bv, bf = self.analyze_param(
                exp_i, exp_f, j,
                grid_points=grid_points,
                scale=scale_per_param[j],
                coarse=coarse,
                do_plot=do_plot
            )
            res[pname] = {
                "values": vals,
                "objectives": fvals,
                "best_value": bv,
                "best_objective": bf,
            }
        return res
