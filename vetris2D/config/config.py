from enum import Enum

class SimulatorType(Enum):
    MPM = "mpm"
    MSD = "msd"

# Choose which simulator to run
SIMULATOR = SimulatorType.MPM

# Simulation hyper‑parameters
TIME_STEP = 1e-4
TOTAL_STEPS = 1000

# Optimizer settings
OPT_LR = 1e-3
OPT_EPOCHS = 50

# Visualization options
RENDER_INTERVAL = 10
