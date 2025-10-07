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



@dataclass
class ExperimentData:
    """Container for simulation time series."""
    time:np.ndarray
    indentation: np.ndarray
    contact_force: np.ndarray






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
    

def load_experiment(csv_path: str, csv_is_mm: bool=False):

    # Load with header
    arr = np.genfromtxt(csv_path,delimiter=",",names=True,dtype=float,encoding="utf-8")
    
    if arr.dtype.names != ("time", "indentation", "contact_force"):
        raise ValueError(
            f"CSV must have EXACT header: 'time,indentation,contact_force'. "
            f"Found: {arr.dtype.names}"
        )

    # Extract columns
    time   = np.asarray(arr["time"], dtype=float)
    indent = np.asarray(arr["indentation"], dtype=float)
    force  = np.asarray(arr["contact_force"], dtype=float)


    if csv_is_mm:
        indent *= 1e-3

    data = ExperimentData(time=time,indentation=indent,contact_force=force)
    return data




exp_data = load_experiment("data/real/loop_1_filtered.csv", csv_is_mm=True)

print(exp_data.indentation)