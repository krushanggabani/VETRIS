import os
import numpy as np

from .config import CalibConfig
from .utils import OutputPaths
from .utils import load_experiment
from .utils import CFLPolicy
from .engine_runner import EngineRunner
from .losses import HysteresisLoss
from .objective import Objective
from .plotting import Plotter
from .optim_random import RandomSearch
from .optim_scipy import ScipyRefiner
# from .optim_cma import CMAESOptimizer
from .optim_bayes import BayesianTPE
from .param_sensitivity import ParamSensitivity


def save_arrays(final_dir, exp_i, exp_f, sim_i, sim_f, x_star):
    np.savetxt(os.path.join(final_dir, "experiment_curve.csv"),
               np.c_[exp_i, exp_f], delimiter=",", fmt="%.8e",
               header="indentation_m,force_N")
    np.savetxt(os.path.join(final_dir, "sim_curve.csv"),
               np.c_[sim_i, sim_f], delimiter=",", fmt="%.8e",
               header="indentation_m,force_N")
    np.savetxt(os.path.join(final_dir, "optimal_params.csv"),
               np.asarray(x_star)[None,:], delimiter=",", fmt="%.8e",
               header="E,nu,eta_shear,eta_bulk,rate_k,rate_n")

def run_pipeline(cfg: CalibConfig):
    # 1) paths + data
    paths = OutputPaths.make(cfg.file_location, cfg.exp_subdir)
    exp_data = load_experiment(cfg.experiment_csv, csv_is_mm=cfg.csv_is_mm)

    exp_i = exp_data.indentation
    exp_f = exp_data.contact_force
    print(f"[data] {exp_i.size} points; indent [{exp_i.min():.4e}, {exp_i.max():.4e}] m, force [{exp_f.min():.4e}, {exp_f.max():.4e}] N")

    # 2) engine + objective
    cfl = CFLPolicy(cfg.rho, cfg.n_grid_dt, cfg.cfl_number, cfg.dt_min, cfg.dt_max)
    runner = EngineRunner(cfg.massager_type, cfg.override_time_period, cfl)
    plotter = Plotter(paths.runs_dir, paths.final_dir)

    loss = HysteresisLoss(weights=cfg.weights)
    objective = Objective(loss)

    # 3) optimization

    # param_names = ["E", "nu", "eta_s", "eta_b", "rate_k", "rate_n"]  # same order as your x
    # # param_names = ["E", "nu", "eta_s", "eta_b"]  # same order as your x
    # sens = ParamSensitivity(
    #     engine_runner=runner,
    #     objective=objective,
    #     plotter=plotter,
    #     bounds=cfg.bounds,             
    #     x_ref=cfg.x0,              
    #     param_names=param_names,
    #     out_dir=os.path.join(plotter.final_dir, "sensitivity")
    # )

    # results = sens.analyze_all(exp_i, exp_f, grid_points=21, scale_per_param=["log", "linear", "log", "log", "linear", "linear"])

    bayes = BayesianTPE(runner, objective, plotter, cfg.bounds, cfg.x0)
    x_best, f_best = bayes.run_infinite(exp_i, exp_f, coarse=True)

    # x_best, f_best = ref.refine(x_best, exp_i, exp_f, coarse=False, maxiter=30)


    # x_best = np.array(cfg.x0, float)
    # f_best = float("inf")

    # if cfg.do_coarse:
    #     rs = RandomSearch(runner, objective, plotter, cfg.bounds, cfg.x0, sigma=cfg.coarse_sigma)
    #     x_best, f_best = rs.run(exp_i, exp_f, trials=cfg.coarse_trials)
    #     print(f"\n[coarse] best f={f_best:.4e}\n        x={x_best}")

    # # 4) scipy refine (fine)
    # if cfg.do_fine:
    #     ref = ScipyRefiner(runner, objective, plotter, cfg.bounds)
    #     x_best, f_best = ref.refine(x_best, exp_i, exp_f)



    # # 5) final run + persist
    # si, sf, _ = runner.run(x_best, coarse=False)
    # save_arrays(paths.final_dir, exp_i, exp_f, si, sf, x_best)
    # plotter.plot_curves(exp_i, exp_f, si, sf,
    #                     title=f"Final fit  f={f_best:.3e}",
    #                     out_name="final_fit.png")
    # print("\n[done] best params:")
    # print(f"  youngs_modulus   = {x_best[0]:.6e}")
    # print(f"  poisson_ratio    = {x_best[1]:.6f}")
    # print(f"  shear_viscosity  = {x_best[2]:.6e}")
    # print(f"  bulk_viscosity   = {x_best[3]:.6e}")
    # print(f"  rate_k           = {x_best[4]:.6f}")
    # print(f"  rate_n           = {x_best[5]:.6f}")
    # print(f"  objective value  = {f_best:.6e}")
    # print(f"Saved: {paths.final_dir}/{{progress.png, best_so_f ar.png, final_fit.png, experiment_curve.csv, sim_curve.csv, optimal_params.csv}} + per-run images in {paths.runs_dir}/")
