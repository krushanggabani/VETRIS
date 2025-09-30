from typing import Tuple
import numpy as np

try:
    from scipy.optimize import minimize
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

def clip_to_bounds(x: np.ndarray, bounds):
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    return np.minimum(np.maximum(x, lo), hi)

class ScipyRefiner:
    def __init__(self, engine_runner, objective, plotter, bounds):
        self.engine = engine_runner
        self.obj = objective
        self.plot = plotter
        self.bounds = bounds

    def param_str(self, x):
        E, nu, eta_s, eta_b, rate_k, rate_n = x
        return (f"E={E:.3e}, nu={nu:.3f}, eta_s={eta_s:.2e}, "
                f"eta_b={eta_b:.2e}, rate_k={rate_k:.3f}, rate_n={rate_n:.3f}")
    
    def refine(self, x0, exp_i, exp_f) -> Tuple[np.ndarray, float]:
        if not SCIPY_OK:
            si, sf,meta = self.engine.run(x0, coarse=False)
            f, _, _ = self.obj.evaluate(x0, exp_i, exp_f, si, sf,meta)

            return x0, f

        bnds = self.bounds
        eval_idx = {"i": 0}

        def fun(x):
            x = clip_to_bounds(np.asarray(x, float), bnds)
            si, sf,meta = self.engine.run(x, coarse=False)


            f, _, _ = self.obj.evaluate(x, exp_i, exp_f, si, sf,meta)
            self.plot.log_progress(f); self.plot.plot_progress()
            eval_idx["i"] += 1
            self.plot.plot_curves(exp_i, exp_f, si, sf,
                                  title=f"SciPy eval {eval_idx['i']}  f={f:.3e}",
                                  out_name=f"run_scipy_{eval_idx['i']:03d}.png",
                                  annotate=self.param_str(x),
                                  to_runs=True)
            return f

        print("\n[scipy] L-BFGS-B refine...")
        res = minimize(fun, x0, method="L-BFGS-B", bounds=bnds, options=dict(maxiter=18, ftol=1e-6))
        x = clip_to_bounds(res.x, bnds)
        f = float(res.fun)
        print(f"[scipy] status={res.status}  f={f:.4e}  x={x}")
        return x, f
