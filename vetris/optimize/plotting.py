import os, csv, numpy as np
import matplotlib.pyplot as plt
from typing import Sequence
from .utils import ExperimentData

class Plotter:
    def __init__(self, runs_dir: str, final_dir: str):
        self.runs_dir = runs_dir
        self.final_dir= final_dir
        self.fx_hist = []
        self.best_hist = []
        self.param_log = os.path.join(self.runs_dir, "param_log.csv")

        # init CSV header
        if not os.path.exists(self.param_log):
            with open(self.param_log, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "trial", "fval", "finished", "progress", "reason",
                    "E", "nu", "eta_s", "eta_b", "rate_k", "rate_n",
                    "rmse", "rmse_load", "rmse_unload",
                    "n_load", "n_unload",
                    "d_peak", "d_slope", "d_area",
                    "slope_exp_head", "slope_exp_tail",
                    "slope_sim_head", "slope_sim_tail",
                    "area_exp", "area_sim"
                ])

    def log_progress(self, f: float):
        self.fx_hist.append(f)
        best = min(self.best_hist[-1], f) if self.best_hist else f
        self.best_hist.append(best)

    def plot_progress(self):
        out = os.path.join(self.final_dir, "progress.png")
        xs = np.arange(1, len(self.fx_hist)+1)
        plt.figure()
        plt.plot(xs, self.fx_hist, marker='o', label="f(x_k)")
        plt.plot(xs, self.best_hist, marker='o', linestyle='--', label="best so far")
        plt.yscale('log'); plt.xlabel("trial"); plt.ylabel("objective")
        plt.legend(); plt.grid(True, which='both'); plt.tight_layout()
        plt.savefig(out); plt.close()

    def plot_curves(self, exp_data, sim_data, title: str, out_name: str, annotate: str=None, to_runs=False):
        
        exp_i, exp_f = exp_data.indentation, exp_data.contact_force, 
        sim_i, sim_f = sim_data.indentation, sim_data.contact_force
        out_dir = self.runs_dir if to_runs else self.final_dir
        out = os.path.join(out_dir, out_name)
        plt.figure()
        plt.scatter(exp_i, exp_f, label="Experiment")
        if sim_i.size > 0:
            plt.scatter(sim_i, sim_f, label="Simulation")
        plt.xlabel("Indentation (m)"); plt.ylabel("Force (N)")
        plt.title(title)
        if annotate:
            plt.suptitle(annotate, fontsize=9, y=0.98)
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(out); plt.close()


    def log_trial(self, trial: int, x: Sequence[float], fval: float, comps:dict, meta: dict):
        """Append scalar trial record."""


        with open(self.param_log, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                trial,
                fval,
                meta.get("finished", None),
                meta.get("progress", None),
                meta.get("reason", None),
                *[f"{float(v):.6e}" for v in x],  # parameters
                f"{comps.get('rmse', ''):.6e}" if 'rmse' in comps else "",
                f"{comps.get('rmse_load', ''):.6e}" if 'rmse_load' in comps else "",
                f"{comps.get('rmse_unload', ''):.6e}" if 'rmse_unload' in comps else "",
                comps.get("n_load", ""),
                comps.get("n_unload", ""),
                f"{comps.get('d_peak', ''):.6e}" if 'd_peak' in comps else "",
                f"{comps.get('d_slope', ''):.6e}" if 'd_slope' in comps else "",
                f"{comps.get('d_area', ''):.6e}" if 'd_area' in comps else "",
                f"{comps.get('slope_exp_head', ''):.6e}" if 'slope_exp_head' in comps else "",
                f"{comps.get('slope_exp_tail', ''):.6e}" if 'slope_exp_tail' in comps else "",
                f"{comps.get('slope_sim_head', ''):.6e}" if 'slope_sim_head' in comps else "",
                f"{comps.get('slope_sim_tail', ''):.6e}" if 'slope_sim_tail' in comps else "",
                f"{comps.get('area_exp', ''):.6e}" if 'area_exp' in comps else "",
                f"{comps.get('area_sim', ''):.6e}" if 'area_sim' in comps else "",
            ])
            
    def log_curve(self, trial: int, sim_data:ExperimentData):
        """Save indentation/force curve as CSV for this trial."""
        out = os.path.join(self.runs_dir, f"curve_{trial:03d}.csv")
        np.savetxt(out, np.c_[sim_data.time,sim_data.indentation, sim_data.contact_force], delimiter=",", fmt="%.8e", header="time_s,indentation_m,force_N")