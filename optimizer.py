from vetris.engine.sim import Simulation
from vetris.config.default_config import CONFIG


CONFIG.engine.massager.type = "straight"  # "straight" or "dual_arm"



CONFIG.engine.mpm.youngs_modulus = 5.65e4
CONFIG.engine.mpm.poisson_ratio  = 0.185



def main():
    sim = Simulation(cfg=CONFIG)
    sim.run()




if __name__ == "__main__":
    main()