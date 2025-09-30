import numpy as np
from typing import Sequence, Tuple, Callable

def clip_to_bounds(x: np.ndarray, bounds) -> np.ndarray:
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    return np.minimum(np.maximum(x, lo), hi)

class RandomSearch:
    def __init__(self, engine_runner, objective, plotter, bounds, x0, sigma=0.6, seed=12):
        self.engine = engine_runner
        self.obj = objective
        self.plot = plotter
        self.bounds = bounds
        self.x0 = np.array(x0, float)
        self.sigma = sigma
        self.rng = np.random.default_rng()

    def param_str(self, x):
        E, nu, eta_s, eta_b, rate_k, rate_n = x
        return (f"E={E:.3e}, nu={nu:.3f}, eta_s={eta_s:.2e}, "
                f"eta_b={eta_b:.2e}, rate_k={rate_k:.3f}, rate_n={rate_n:.3f}")

    def run(self, exp_i, exp_f, trials=24) -> Tuple[np.ndarray, float]:
        best_x = clip_to_bounds(self.x0.copy(), self.bounds)
        si0, sf0,meta = self.engine.run(best_x, coarse=True)
        f0, _, _ = self.obj.evaluate(best_x, exp_i, exp_f, si0, sf0,meta)
        best_f = f0

        self.plot.log_progress(f0)
        self.plot.plot_progress()
        self.plot.plot_curves(exp_i, exp_f, si0, sf0,
                              title=f"Random k=0  f={f0:.3e} (seed)",
                              out_name="run_random_000.png", annotate=self.param_str(best_x),
                              to_runs=True)
        self.plot.plot_curves(exp_i, exp_f, si0, sf0,
                              title=f"Best-so-far f={best_f:.3e}",
                              out_name="best_so_far.png", annotate=self.param_str(best_x))

        for k in range(trials):
            # multiplicative / additive jitters similar to your script
            x = self.x0.copy()
            x[0] *= 10**self.rng.normal(0.0, self.sigma)      # E
            x[1] *= self.rng.normal(1.0, 0.2)                 # nu
            x[2] *= 10**self.rng.normal(0.0, self.sigma)      # eta_s
            x[3] *= 10**self.rng.normal(0.0, self.sigma)      # eta_b
            x[4]  = np.clip(max(0.0, self.rng.normal(0.5, 0.5)), self.bounds[4][0], self.bounds[4][1])  # rate_k
            x[5] *= self.rng.normal(1.0, 0.3)                 # rate_n
            x = clip_to_bounds(x, self.bounds)

            si, sf,meta = self.engine.run(x, coarse=True)
            f, _, _ = self.obj.evaluate(x, exp_i, exp_f, si, sf,meta)

            self.plot.log_progress(f)
            if f < best_f:
                best_x, best_f = x, f
                self.plot.plot_curves(exp_i, exp_f, si, sf,
                                      title=f"Best-so-far f={best_f:.3e}",
                                      out_name="best_so_far.png", annotate=self.param_str(best_x))
            self.plot.plot_curves(exp_i, exp_f, si, sf,
                                  title=f"Random k={k+1}  f={f:.3e}",
                                  out_name=f"run_random_{k+1:03d}.png",
                                  annotate=self.param_str(x),
                                  to_runs=True)
            self.plot.plot_progress()
            print(f"[random] k={k+1:02d}/{trials:02d}  f={f:.4e}  best={best_f:.4e}  x={x}")
        return best_x, best_f
