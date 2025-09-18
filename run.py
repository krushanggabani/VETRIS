from vetris.engine.sim import Simulation
from vetris.config.default_config import CONFIG

def main():
    sim = Simulation(cfg=CONFIG)
    sim.run()




if __name__ == "__main__":
    main()