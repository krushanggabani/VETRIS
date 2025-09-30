from typing import Tuple, Sequence, Optional, Dict, Any
import numpy as np
import taichi as ti

from vetris.config.default_config import CONFIG
from vetris.engine.sim import Simulation

class EngineRunner:
    """Thin adapter around your Taichi engine to produce (indentation_m, force_N) curves,
    and report whether the simulation finished."""
    def __init__(self, massager_type: str, override_time_period: Optional[float], cfl_policy):
        self.massager_type = massager_type
        self.override_time_period = override_time_period
        self.cfl_policy = cfl_policy

    def _apply_params_to_config(self, params: Sequence[float], dt: float):
        E, nu, eta_s, eta_b, rate_k, rate_n = params
        CONFIG.engine.massager.type       = self.massager_type
        CONFIG.engine.dt                  = float(dt)
        CONFIG.engine.mpm.youngs_modulus  = float(E)
        CONFIG.engine.mpm.poisson_ratio   = float(nu)
        CONFIG.engine.mpm.shear_viscosity = float(eta_s)
        CONFIG.engine.mpm.bulk_viscosity  = float(eta_b)
        CONFIG.engine.mpm.rate_k          = float(rate_k)
        CONFIG.engine.mpm.rate_n          = float(rate_n)

    def run(
        self,
        params: Sequence[float],
        coarse: bool=False,
        max_outer_steps: int = 20000,      # hard safety cap
        stagnation_tol: int = 5            # how many non-increasing time samples before bailing
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Returns (si, sf, meta) where meta contains:
           - finished: bool
           - progress: float in [0,1] (time / stop_time, clamped)
           - time: final engine time
           - stop_time: planned stop time
           - steps: number of outer while iterations taken
           - reason: 'ok' | 'time_stagnation' | 'max_steps' | 'nan' | 'coarse_cap'
        """
        E, nu, *_ = params
        DT = self.cfl_policy.compute_stable_dt(E, nu)

        ti.reset()
        ti.init(arch=ti.gpu)

        self._apply_params_to_config(params, DT)
        sim = Simulation(cfg=CONFIG)

        stop_time = float(sim.engine.massager.massager.Time_period) \
                    if self.override_time_period is None else float(self.override_time_period)

        sim_indent, sim_force = [], []
        last_t = -1.0
        non_increasing = 0
        reason = "ok"

        steps = 0
        while float(sim.engine.massager.massager.time_t) < stop_time:
            steps += 1
            if steps > max_outer_steps:
                reason = "max_steps"
                break

            for _ in range(15):
                sim.engine.run()

            # headless-safe render
            try:
                sim.renderer.render(sim.engine)
            except Exception:
                pass

            state = sim.engine.get_state()
            t     = float(state["time"])
            # NOTE: keep your chosen observable; here you used max_eqv_strain_contact
            indent_m = float(state["deformation"]["max_eqv_strain_contact"]) * 1e-3

            cf = np.asarray(state["contact_force"])
            force_N = float(cf) if cf.ndim == 0 else float(np.linalg.norm(cf.reshape(-1)))

            # Check for NaNs / infs (engine blew up)
            if not np.isfinite(indent_m) or not np.isfinite(force_N) or not np.isfinite(t):
                reason = "nan"
                break

            sim_indent.append(indent_m)
            sim_force.append(force_N)

            # time stagnation check
            if t <= last_t:
                non_increasing += 1
                if non_increasing >= stagnation_tol:
                    reason = "time_stagnation"
                    break
            else:
                non_increasing = 0
            last_t = t

            # coarse early cap
            if coarse and len(sim_indent) > 2000:
                reason = "coarse_cap"
                break

        # Determine completion/progress
        final_time = max(float(last_t), float(sim.engine.massager.massager.time_t))
        progress = float(np.clip(final_time / max(stop_time, 1e-9), 0.0, 1.0))
        finished = (progress >= 1.0) and (reason in ("ok", "coarse_cap"))

        si = np.asarray(sim_indent, dtype=float)
        sf = np.asarray(sim_force,  dtype=float)

        meta = dict(
            finished=bool(finished),
            progress=progress,
            time=final_time,
            stop_time=float(stop_time),
            steps=int(steps),
            reason=reason,
        )
        return si, sf, meta
