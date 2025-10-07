import numpy as np
from typing import Sequence, Tuple, Dict

def regularization(x: Sequence[float]) -> float:
    E, nu, eta_s, eta_b, rate_k, rate_n = x
    return (
        1e-6 * (np.log(max(E, 1.0))**2) +
        1e-5 * ((nu - 0.3)**2) +
        1e-10 * (eta_s**2 + 0.1*eta_b**2) +
        1e-6 * ((rate_k)**2 + (rate_n - 1.0)**2)
    )

# class Objective:
#     """Wraps (loss + regularization + coverage penalty)."""
#     def __init__(self, loss_callable):
#         self.loss = loss_callable

#     def evaluate(self, x, exp_i, exp_f, sim_i, sim_f) -> Tuple[float, Dict, int]:
#         err, comps, npts = self.loss(exp_i, exp_f, sim_i, sim_f)
#         if not np.isfinite(err):
#             return 1e12, comps, npts
#         coverage_penalty = 1e-2 / max(npts, 1)
#         f = float(err + regularization(x) + coverage_penalty)
#         return f, comps, npts

def unfinished_penalty(base_loss: float, meta: dict, scale: float = 1e2, p: float = 2.0) -> float:
    """
    Penalize runs that didn’t finish. Penalty = scale * (1 - progress)^p.
    - progress in [0,1]
    - scale controls magnitude relative to your loss.
    - p>1 makes the penalty softer when near-finished, harsher when early-exit.
    """
    if meta is None:
        return base_loss
    prog = float(meta.get("progress", 0.0))
    if meta.get("finished", False):
        return base_loss
    

    return base_loss + scale * (1.0 - prog) ** p



class Objective:
    def __init__(self, loss_callable, penalty_scale: float = 1e4, penalty_p: float = 1.0):
        self.loss = loss_callable
        self.penalty_scale = penalty_scale
        self.penalty_p = penalty_p


    def evaluate(self, x, exp_data, sim_data, meta=None):
        


        err, comps, npts = self.loss(exp_data, sim_data)

        if not np.isfinite(err):
            err = 1e12
        # coverage_penalty = 1e-2 / max(npts, 1)
        coverage_penalty = 0
        f = float(err + coverage_penalty)
        # f += regularization(x)
        f = unfinished_penalty(f, meta, scale=self.penalty_scale, p=self.penalty_p)
        return f, comps, npts

