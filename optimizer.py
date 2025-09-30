from vetris.engine.sim import Simulation
from vetris.config.default_config import CONFIG
from vetris.io.plotter import Plotter

import math
import numpy as np

CONFIG.engine.massager.type = "straight"  # "straight" or "dual_arm"



CONFIG.engine.mpm.youngs_modulus = 5.65e4
CONFIG.engine.mpm.poisson_ratio  = 0.185
CONFIG.engine.mpm.shear_viscosity = 150.0
CONFIG.engine.mpm.bulk_viscosity = 50
CONFIG.engine.mpm.rate_k = 0.0
CONFIG.engine.mpm.rate_n = 1.0
CONFIG.engine.dt  = 2e-4


youngs_modulus   = 1.721394e+04
poisson_ratio    = 0.204221
shear_viscosity  = 2.612192e+01
bulk_viscosity   = 2.532133e+00
rate_k           = 0.717367
rate_n           = 1.519987



# Update CONFIG with these parameter values for consistency
CONFIG.engine.mpm.youngs_modulus   = youngs_modulus
CONFIG.engine.mpm.poisson_ratio    = poisson_ratio
CONFIG.engine.mpm.shear_viscosity  = shear_viscosity
CONFIG.engine.mpm.bulk_viscosity   = bulk_viscosity
CONFIG.engine.mpm.rate_k           = rate_k
CONFIG.engine.mpm.rate_n           = rate_n



# CFL params for dt computation
RHO         = 1.0
N_GRID_DT   = 128      # -> dx = 1/128
CFL_NUMBER  = 0.38     # 0.35–0.40 good; lower if unstable
DT_MIN      = 5e-7
DT_MAX      = 5e-4


def compute_stable_dt(E, nu, rho=RHO, n_grid=N_GRID_DT, cfl=CFL_NUMBER):
    """Δt ≤ CFL * Δx / sqrt((λ + 2μ)/ρ)."""
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    c = math.sqrt(max((lam + 2.0 * mu) / rho, 1e-20))
    dx = 1.0 / float(n_grid)
    dt = cfl * dx / c
    return float(np.clip(dt, DT_MIN, DT_MAX))



DT = compute_stable_dt(CONFIG.engine.mpm.youngs_modulus,CONFIG.engine.mpm.poisson_ratio)
CONFIG.engine.dt  = DT

print(DT)
def main():
    sim = Simulation(cfg=CONFIG)
    sim.run()

   


# if __name__ == "__main__":

#     # main()

#     p= Plotter("data/logs/simulation_logs.csv")

#     sim_i,sim_f,t = p._get_xyt()

#     exp_i,exp_f = p._get_raw()

#     # # plain scatter
#     # p.plot(show=True)


#     # show movement as a line + small arrows every ~10 steps
#     # p.plot_path(step=1, arrows_every=10, show=True)


#     # animation (mp4 or gif)
#     # p.animate_force_strain("strain_vs_force.mp4", fps=24, tail=80)



from vetris.optimize.losses import HysteresisLoss
from vetris.optimize.objective import Objective

p= Plotter("data/logs/simulation_logs.csv")

sim_i,sim_f,t = p._get_xyt()
exp_i,exp_f = p._get_raw()

loss = HysteresisLoss(weights=(10.0, 0.5, 100, 50.0))
objective = Objective(loss)


f, com, _ = objective.evaluate(p, exp_i, exp_f, sim_i, sim_f,meta="fine")

