from typing import Tuple, List
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
        self._trial_counter = 0  # keep a consistent trial index

    def _eval(self, x, exp_i, exp_f, coarse=True, trial_id=None):
        si, sf, meta = self.engine.run(x, coarse=coarse)
        f, _, _ = self.obj.evaluate(x, exp_i, exp_f, si, sf, meta=meta)

        # plotting (optional)
        self.plot.log_progress(f)
        self.plot.plot_progress()
        self.plot.plot_curves(exp_i, exp_f, si, sf,
                              title=f"Bayes eval  f={f:.3e}",
                              out_name=f"run_bayes_eval_{trial_id:03d}.png",
                              to_runs=True)

        # logging (scalar + curve)
        if trial_id is not None:
            self.plot.log_trial(trial_id, x, f, meta)
            self.plot.log_curve(trial_id, si, sf)

        return float(f)

    def run(self, exp_i, exp_f, coarse=True) -> Tuple[np.ndarray, float]:
        if not OPTUNA_OK:
            # Simple random fallback
            best_x = _clip(self.x0.copy(), self.bounds)
            best_f = self._eval(best_x, exp_i, exp_f, coarse=coarse, trial_id=0)
            for k in range(1, self.n_trials+1):
                lo = np.array([b[0] for b in self.bounds], float)
                hi = np.array([b[1] for b in self.bounds], float)
                x = lo + self.rng.random(lo.shape) * (hi - lo)
                f = self._eval(x, exp_i, exp_f, coarse=coarse, trial_id=k)
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
