# calibrate_ls_plot.py
import numpy as np
import taichi as ti
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from vetris.config.default_config import CONFIG
from vetris.engine.sim import Simulation

# -------- user inputs --------
EXPERIMENT_CSV = "data/real/force_indentation.csv"   # two cols: indentation_m, force_N
MASSAGER_TYPE  = "straight"                          # or "dual_arm"
DT             = 1e-4
CSV_IS_MM      = False                               # set True if indentation is in mm
EXP_MAX_POINTS = 1500                                 # set None to disable downsample

# parameter bounds: [E, nu]
BOUNDS_LO = np.array([1e3, 0.01], dtype=float)
BOUNDS_HI = np.array([5e6, 0.49], dtype=float)

# starting guess
X0 = np.array([5.65e4, 0.185], dtype=float)
# --------------------------------


def load_experiment(path, csv_is_mm=False, max_points=None):
    raw = np.loadtxt(path, delimiter=",")
    ind = raw[:, 0].astype(float)
    frc = raw[:, 1].astype(float)
    if csv_is_mm:
        ind *= 1e-3  # mm -> m
    order = np.argsort(ind)
    ind, frc = ind[order], frc[order]
    if max_points is not None and ind.size > max_points:
        idx = np.linspace(0, ind.size - 1, max_points).round().astype(int)
        ind, frc = ind[idx], frc[idx]
    return ind, frc


def run_sim_and_curve(E, nu):
    ti.reset()
    ti.init(arch=ti.gpu)

    CONFIG.engine.massager.type = MASSAGER_TYPE
    CONFIG.engine.dt = DT
    CONFIG.engine.mpm.youngs_modulus = float(E)
    CONFIG.engine.mpm.poisson_ratio  = float(nu)

    sim = Simulation(cfg=CONFIG)
    stop_time = float(sim.engine.massager.massager.Time_period)

    sim_ind, sim_frc = [], []
    last_t = -1.0

    while float(sim.engine.massager.massager.time_t) < stop_time:
        for _ in range(15):
            sim.engine.run()

        st = sim.engine.get_state()
        t  = float(st["time"])

        ind_m = float(st["deformation"]["indentation"]) * 1e-3  # mm -> m
        cf = np.asarray(st["contact_force"])
        if cf.ndim == 0:
            fN = float(cf)
        else:
            fN = float(np.linalg.norm(cf.reshape(-1)))

        sim_ind.append(ind_m)
        sim_frc.append(fN)

        if t <= last_t:
            break
        last_t = t

    sim_ind = np.asarray(sim_ind, dtype=float)
    sim_frc = np.asarray(sim_frc, dtype=float)
    if sim_ind.size == 0:
        return sim_ind, sim_frc
    order = np.argsort(sim_ind)
    return sim_ind[order], sim_frc[order]


def residuals(x, exp_ind, exp_frc):
    E = float(np.clip(x[0], BOUNDS_LO[0], BOUNDS_HI[0]))
    nu = float(np.clip(x[1], BOUNDS_LO[1], BOUNDS_HI[1]))

    sim_ind, sim_frc = run_sim_and_curve(E, nu)
    if sim_ind.size < 5:
        return np.ones_like(exp_frc) * 1e6

    lo = max(exp_ind.min(), sim_ind.min())
    hi = min(exp_ind.max(), sim_ind.max())
    mask = (exp_ind >= lo) & (exp_ind <= hi)
    if not np.any(mask):
        return np.ones_like(exp_frc) * 1e6

    sim_i, uniq = np.unique(sim_ind, return_index=True)
    sim_f = sim_frc[uniq]

    sim_on_exp = np.interp(exp_ind[mask], sim_i, sim_f)
    return sim_on_exp - exp_frc[mask]


def main():
    exp_ind, exp_frc = load_experiment(EXPERIMENT_CSV, csv_is_mm=CSV_IS_MM, max_points=EXP_MAX_POINTS)
    print(f"[data] {exp_ind.size} points loaded")

    res = least_squares(
        fun=lambda x: residuals(x, exp_ind, exp_frc),
        x0=X0,
        bounds=(BOUNDS_LO, BOUNDS_HI),
        max_nfev=100
    )

    E_opt, nu_opt = res.x
    print(f"\n[optimal] E={E_opt:.3e}, nu={nu_opt:.4f}, cost={0.5*np.sum(res.fun**2):.3e}")

    sim_ind, sim_frc = run_sim_and_curve(E_opt, nu_opt)

    # --- save ---
    np.savetxt("calib_out_experiment.csv", np.c_[exp_ind, exp_frc], delimiter=",",
               header="indentation_m,force_N", fmt="%.8e")
    np.savetxt("calib_out_sim.csv", np.c_[sim_ind, sim_frc], delimiter=",",
               header="indentation_m,force_N", fmt="%.8e")
    np.savetxt("calib_out_params.csv", [[E_opt, nu_opt]], delimiter=",",
               header="E,nu", fmt="%.8e")

    # --- plot ---
    plt.figure(figsize=(6,4))
    plt.plot(exp_ind*1e3, exp_frc, 'o', label="Experiment", alpha=0.7)   # mm on x-axis
    plt.plot(sim_ind*1e3, sim_frc, '-', label="Simulation")
    plt.xlabel("Indentation [mm]")
    plt.ylabel("Force [N]")
    plt.title("Calibration Result")
    plt.legend()
    plt.tight_layout()
    plt.savefig("calib_out_plot.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
