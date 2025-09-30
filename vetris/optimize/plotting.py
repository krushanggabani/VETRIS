import os, csv, numpy as np
import matplotlib.pyplot as plt
from typing import Sequence

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
                    "E", "nu", "eta_s", "eta_b", "rate_k", "rate_n"
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

    def plot_curves(self, exp_i, exp_f, sim_i, sim_f, title: str, out_name: str, annotate: str=None, to_runs=False):
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


    def log_trial(self, trial: int, x: Sequence[float], fval: float, meta: dict):
        """Append scalar trial record."""
        with open(self.param_log, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                trial, fval, meta.get("finished", None), meta.get("progress", None), meta.get("reason", None),
                *[f"{v:.6e}" for v in x]
            ])

    def log_curve(self, trial: int, si: np.ndarray, sf: np.ndarray):
        """Save indentation/force curve as CSV for this trial."""
        out = os.path.join(self.runs_dir, f"curve_{trial:03d}.csv")
        np.savetxt(out, np.c_[si, sf], delimiter=",", fmt="%.8e", header="indentation_m,force_N")