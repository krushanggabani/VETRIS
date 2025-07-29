from config import SIMULATOR, TIME_STEP, TOTAL_STEPS, OPT_LR, OPT_EPOCHS, RENDER_INTERVAL
from simulators.mpm import MpmSimulator
from simulators.msd import MsdSimulator
from optimizer.optimizer import SimulatorOptimizer
from visualizer.visualizer import SimulatorVisualizer

def build_simulator():
    if SIMULATOR.value == "mpm":
        return MpmSimulator(time_step=TIME_STEP, material_params={ /*…*/ })
    else:
        return MsdSimulator(time_step=TIME_STEP, msd_params={ /*…*/ })

def main():
    sim = build_simulator()

    # Option A: just run visualization
    viz = SimulatorVisualizer(sim, render_interval=RENDER_INTERVAL)
    viz.render()

    # Option B: run training
    # opt = SimulatorOptimizer(sim, lr=OPT_LR, epochs=OPT_EPOCHS)
    # opt.train()

if __name__ == "__main__":
    main()
