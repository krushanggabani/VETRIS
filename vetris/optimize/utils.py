import os
from dataclasses import dataclass
import numpy as np
import math


@dataclass
class OutputPaths:
    root: str
    exp_dir: str
    runs_dir: str
    final_dir: str

    @classmethod
    def make(cls, root: str, exp_subdir: str) -> "OutputPaths":
        exp_dir  = os.path.join(root, exp_subdir)
        runs_dir = os.path.join(exp_dir, "runs")
        final_dir= os.path.join(exp_dir, "final")
        for d in (exp_dir, runs_dir, final_dir):
            os.makedirs(d, exist_ok=True)
        return cls(root=root, exp_dir=exp_dir, runs_dir=runs_dir, final_dir=final_dir)



def load_experiment(csv_path: str, csv_is_mm: bool=False, max_points=None):
    raw = np.loadtxt(csv_path, delimiter=",")
    if raw.ndim != 2 or raw.shape[1] < 2:
        raise ValueError("CSV must have at least two columns: indentation, force")
    indent = raw[:,0].astype(float)
    force  = raw[:,1].astype(float)
    if csv_is_mm:
        indent *= 1e-3
    # (Optionally subsample if needed)
    return indent, force



@dataclass
class CFLPolicy:
    rho: float
    n_grid: int
    cfl: float
    dt_min: float
    dt_max: float

    def compute_stable_dt(self, E: float, nu: float) -> float:
        mu  = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        c   = math.sqrt(max((lam + 2.0*mu) / self.rho, 1e-20))
        dx  = 1.0 / float(self.n_grid)
        dt  = self.cfl * dx / c
        return float(np.clip(dt, self.dt_min, self.dt_max))