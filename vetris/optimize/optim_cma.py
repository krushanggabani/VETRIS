from typing import Tuple, List
import numpy as np

try:
    import cma  # pycma
    CMA_OK = True
except Exception:
    CMA_OK = False

def _clip(x: np.ndarray, bounds: List[tuple]) -> np.ndarray:
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    return np.minimum(np.maximum(x, lo), hi)

class CMAESOptimizer:
    """
    CMA-ES with optional random-search warm start.
    Requires `cma` (pycma). If unavailable, it degrades to random search only.
    """
    def __init__(self, engine_runner, objective, plotter,
                 bounds: List[tuple], x0: List[float],
                 warm_trials: int = 32, sigma0_frac: float = 0.25,
                 seed: int = 1234):
        self.engine = engine_runner
        self.obj = objective
        self.plot = plotter
        self.bounds = bounds
        self.x0 = np.array(x0, float)
        self.warm_trials = int(warm_trials)
        self.sigma0_frac = float(sigma0_frac)
        self.rng = np.random.default_rng(seed)

    def _param_str(self, x):
        E, nu, eta_s, eta_b, rate_k, rate_n = x
        return (f"E={E:.3e}, nu={nu:.3f}, eta_s={eta_s:.2e}, "
                f"eta_b={eta_b:.2e}, rate_k={rate_k:.3f}, rate_n={rate_n:.3f}")

    def _random_sample(self) -> np.ndarray:
        lo = np.array([b[0] for b in self.bounds], float)
        hi = np.array([b[1] for b in self.bounds], float)
        # log-uniform for viscous params; normal-ish for nu, rate_n; uniform for others
        x = self.x0.copy()
        # E
        x[0] = 10**self.rng.uniform(np.log10(lo[0]), np.log10(hi[0]))
        # nu
        x[1] = self.rng.uniform(lo[1], hi[1])
        # eta_s, eta_b
        x[2] = 10**self.rng.uniform(np.log10(lo[2]), np.log10(hi[2]))
        x[3] = 10**self.rng.uniform(np.log10(lo[3]), np.log10(hi[3]))
        # rate_k, rate_n
        x[4] = self.rng.uniform(lo[4], hi[4])
        x[5] = self.rng.uniform(lo[5], hi[5])
        return x

    def _eval(self, x, exp_i, exp_f, coarse=True):
        x = _clip(np.asarray(x, float), self.bounds)
        si, sf, meta = self.engine.run(x, coarse=coarse)
        f, _, _ = self.obj.evaluate(x, exp_i, exp_f, si, sf, meta=meta)
        self.plot.log_progress(f); self.plot.plot_progress()
        self.plot.plot_curves(exp_i, exp_f, si, sf,
                              title=f"CMA eval  f={f:.3e}",
                              out_name=f"run_cma_eval.png",
                              to_runs=True)
        return float(f), si, sf

    def run(self, exp_i, exp_f, cma_iters: int = 60, popsize: int = 16, coarse=True) -> Tuple[np.ndarray, float]:
        # Warm start with random search
        best_x = _clip(self.x0.copy(), self.bounds)
        best_f, _, _ = self._eval(best_x, exp_i, exp_f, coarse=coarse)
        self.plot.plot_curves(exp_i, exp_f, *self.engine.run(best_x, coarse=coarse)[:2],
                              title=f"CMA-RS seed f={best_f:.3e}",
                              out_name="cma_rs_seed.png")

        for k in range(self.warm_trials):
            x = self._random_sample()
            f, _, _ = self._eval(x, exp_i, exp_f, coarse=coarse)
            if f < best_f:
                best_x, best_f = x, f

        if not CMA_OK:
            # No CMA available; return best from warm-start
            return best_x, best_f

        # CMA init
        x0 = best_x.copy()
        lo = np.array([b[0] for b in self.bounds], float)
        hi = np.array([b[1] for b in self.bounds], float)
        sigma0 = self.sigma0_frac * np.maximum(1e-8, (hi - lo))
        es = cma.CMAEvolutionStrategy(x0.tolist(), float(np.median(sigma0)), {
            "bounds": [lo.tolist(), hi.tolist()],
            "seed": int(self.rng.integers(0, 2**31-1)),
            "popsize": int(popsize),
            "maxiter": int(cma_iters),
            "verb_log": 0, "verb_disp": 0,
        })

        while not es.stop():
            xs = es.ask()
            fs = []
            for cand in xs:
                f, _, _ = self._eval(cand, exp_i, exp_f, coarse=coarse)
                fs.append(f)
            es.tell(xs, fs)
            es.disp()
        x_cma = _clip(np.array(es.result.xbest, float), self.bounds)
        f_cma, _, _ = self._eval(x_cma, exp_i, exp_f, coarse=coarse)
        return (x_cma, f_cma) if f_cma < best_f else (best_x, best_f)
