from typing import Tuple, List, Optional
import os, csv, time
import numpy as np
import optuna


OPTUNA_OK = True


def _clip(x: np.ndarray, bounds: List[tuple]) -> np.ndarray:
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    return np.minimum(np.maximum(x, lo), hi)


class BayesianTPE:
    """
    Bayesian optimization via Optuna's TPE sampler.
    Falls back to random search if Optuna isn't available.
    """
    def __init__(self, engine_runner, objective, plotter,
                 bounds: List[tuple], x0: List[float],
                 n_trials: int = 80, seed: int = 1234):
        self.engine = engine_runner
        self.obj = objective
        self.plot = plotter
        self.bounds = bounds
        self.x0 = np.array(x0, float)
        self.n_trials = int(n_trials)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self._trial_counter = 0  # global, monotonically increasing across runs
        self._run_counter = 0    # run (study) index starting at 1

        self.best_runs_csv = os.path.join(self.plot.final_dir, "best_runs.csv") 
        if self.best_runs_csv:
            os.makedirs(os.path.dirname(self.best_runs_csv), exist_ok=True)
            if not os.path.exists(self.best_runs_csv):
                with open(self.best_runs_csv, "w", newline="") as f:
                    csv.writer(f).writerow([
                        "run_id", "best_trial_id", "timestamp", "best_value",
                        "E", "nu", "eta_s", "eta_b", "rate_k", "rate_n"
                    ])

    def _log_best_run_row(self, run_id: int, best_trial_id: int, fbest: float, xbest: np.ndarray):
        """Append one row (best-of-run) to best_runs_csv."""
        if not self.best_runs_csv:
            return
        with open(self.best_runs_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                run_id,
                best_trial_id,
                time.strftime("%Y-%m-%d %H:%M:%S"),
                f"{fbest:.8e}",
                *[f"{v:.8e}" for v in xbest.tolist()],
            ])



    def _eval_one_trial(self, x, coarse=True, trial_id=None):
        si, sf, meta = self.engine.run(x, coarse=coarse)

        f, comps, npts = self.obj.evaluate(x, self._exp_i, self._exp_f, si, sf, meta=meta)

        # plotting (optional)
        self.plot.log_progress(f)
        self.plot.plot_progress()
        self.plot.plot_curves(self._exp_i, self._exp_f, si, sf,
                              title=f"Bayes eval  f={f:.3e}",
                              out_name=f"run_bayes_eval_{trial_id:03d}.png",
                              to_runs=True)

        self.plot.log_trial(trial_id, x, f, comps,meta)
        self.plot.log_curve(trial_id, si, sf)

        return float(f)

    def run(self, exp_i, exp_f, coarse=True) -> Tuple[np.ndarray, float]:
        if not OPTUNA_OK:
            # Simple random fallback
            best_x = _clip(self.x0.copy(), self.bounds)
            best_f = self._eval_one_trial(best_x,coarse=coarse, trial_id=0)
            for k in range(1, self.n_trials+1):
                lo = np.array([b[0] for b in self.bounds], float)
                hi = np.array([b[1] for b in self.bounds], float)
                x = lo + self.rng.random(lo.shape) * (hi - lo)
                f = self._eval_one_trial(x, coarse=coarse, trial_id=k)
                if f < best_f:
                    best_x, best_f = x, f
            return best_x, best_f

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)

        def objective_fn(trial: "optuna.trial.Trial"):
            self._trial_counter += 1
            tid = self._trial_counter
            # Suggest on proper scales
            E     = trial.suggest_float("E",     self.bounds[0][0], self.bounds[0][1], log=True)
            nu    = trial.suggest_float("nu",    self.bounds[1][0], self.bounds[1][1])
            eta_s = trial.suggest_float("eta_s", self.bounds[2][0], self.bounds[2][1], log=True)
            eta_b = trial.suggest_float("eta_b", self.bounds[3][0], self.bounds[3][1], log=True)
            rate_k= trial.suggest_float("rate_k",self.bounds[4][0], self.bounds[4][1])
            rate_n= trial.suggest_float("rate_n",self.bounds[5][0], self.bounds[5][1])
            x = np.array([E, nu, eta_s, eta_b, rate_k, rate_n], float)
            return self._eval(x, exp_i, exp_f, coarse=coarse, trial_id=tid)

        study.optimize(objective_fn, n_trials=self.n_trials, show_progress_bar=False)
        xbest = np.array([study.best_params[k] for k in ["E","nu","eta_s","eta_b","rate_k","rate_n"]], float)
        fbest = float(study.best_value)
        return xbest, fbest

    def run_infinite( self,exp_i: np.ndarray,exp_f: np.ndarray,coarse: bool = True,trials_per_run: int = 200,
        patience: int = 30,min_improvement: float = 1e-3,        sampler_name: str = "tpe",  # "cmaes" or "tpe"
    ) -> Tuple[np.ndarray, float]:
        
        """
        Infinite outer loop: restarts new Optuna studies until interrupted.

        - `trials_per_run`: hard cap per run (study)
        - `patience`: stop the run early if no improvement for this many consecutive trials
        - `min_improvement`: required improvement to reset patience
        - `sampler_name`: "cmaes" (good for continuous search) or "tpe"
        """
        self._exp_i = np.asarray(exp_i, float)
        self._exp_f = np.asarray(exp_f, float)

        global_best_x = None
        global_best_f = np.inf

        try:
            while True:
                self._run_counter += 1
                run_id = self._run_counter
                print(f"\n=== BayesianTPE: Run {run_id} started ===")

                # Fresh sampler/study
                if sampler_name.lower() == "cmaes":
                    sampler = optuna.samplers.CmaEsSampler(seed=int(self.rng.integers(1, 2**31 - 1)))
                else:
                    sampler = optuna.samplers.TPESampler(seed=int(self.rng.integers(1, 2**31 - 1)))
                study = optuna.create_study(direction="minimize", sampler=sampler)

                # Map Optuna's per-study trial.number -> our global trial_id
                study_trial_to_global: dict[int, int] = {}

                # Early-stop state for this run
                best_this_run = np.inf
                best_global_tid_this_run = -1
                no_improve = 0

                # Callback uses FrozenTrial.number (per-study)
                def _callback(st: "optuna.study.Study", t: "optuna.trial.FrozenTrial"):
                    nonlocal best_this_run, best_global_tid_this_run, no_improve
                    val = float(t.value)
                    # our global id for this per-study trial.number:
                    gtid = study_trial_to_global.get(t.number, -1)
                    if val + min_improvement < best_this_run:
                        best_this_run = val
                        best_global_tid_this_run = gtid
                        no_improve = 0
                    else:
                        no_improve += 1
                        if no_improve >= patience:
                            st.stop()  # early stop this run

                # Objective: suggest params, evaluate, and register mapping
                def _objective(trial: "optuna.trial.Trial") -> float:
                    self._trial_counter += 1
                    global_tid = self._trial_counter
                    study_trial_to_global[trial.number] = global_tid

                    # Suggest on appropriate scales
                    E      = trial.suggest_float("E",      self.bounds[0][0], self.bounds[0][1], log=True)
                    nu     = trial.suggest_float("nu",     self.bounds[1][0], self.bounds[1][1])
                    eta_s  = trial.suggest_float("eta_s",  self.bounds[2][0], self.bounds[2][1], log=True)
                    eta_b  = trial.suggest_float("eta_b",  self.bounds[3][0], self.bounds[3][1], log=True)
                    rate_k = trial.suggest_float("rate_k", self.bounds[4][0], self.bounds[4][1])
                    rate_n = trial.suggest_float("rate_n", self.bounds[5][0], self.bounds[5][1])

                    x = np.array([E, nu, eta_s, eta_b, rate_k, rate_n], float)
                    return self._eval_one_trial(x, coarse=coarse, trial_id=global_tid)

                # Run this study
                study.optimize(
                    _objective,
                    n_trials=int(trials_per_run),
                    callbacks=[_callback],
                    show_progress_bar=False
                )

                # Record best-of-run
                if len(study.trials) > 0 and study.best_value is not None:
                    xbest = np.array([
                        study.best_params["E"],
                        study.best_params["nu"],
                        study.best_params["eta_s"],
                        study.best_params["eta_b"],
                        study.best_params["rate_k"],
                        study.best_params["rate_n"],
                    ], float)
                    fbest = float(study.best_value)

                    # Write per-run best row with our global trial id
                    self._log_best_run_row(run_id, best_global_tid_this_run, fbest, xbest)
                    print(f"[Run {run_id}] best f={fbest:.6e}, global_tid={best_global_tid_this_run}, x={xbest}")

                    if fbest < global_best_f:
                        global_best_x, global_best_f = xbest, fbest

                early = (len(study.trials) < trials_per_run)
                print(f"=== BayesianTPE: Run {run_id} ended (early_stop={early}) ===")

        except KeyboardInterrupt:
            print("\n[BayesianTPE] Interrupted by user. Returning global best.")

        if global_best_x is None:
            # Edge case: nothing ran; evaluate x0 once
            self._trial_counter += 1
            f0 = self._eval_one_trial(self.x0, coarse=coarse, trial_id=self._trial_counter)
            return self.x0, f0

        return global_best_x, global_best_f