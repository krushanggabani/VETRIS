import taichi as ti
import numpy as np
import time

from vetris.vis.render import Renderer
from vetris.engine.mpm import mpmengine
from vetris.io.recorder import Recorder
from vetris.io.logger import Logger

ti.init(arch=ti.gpu, debug=False, fast_math=True, device_memory_GB=9)

class Simulation:
    def __init__(self, cfg):

        self.cfg      = cfg       
        self.recorder = Recorder(cfg)
        # self.renderer = Renderer(cfg)
        self.engine  = mpmengine(cfg)
        self.logger  = Logger(cfg)
        self.headless = True
        # print("Simulation initialized")


    def run(self):
        print("Running simulation...")
        if self.headless:

            # while self.renderer.gui.running and self.engine.massager.massager.time_t < self.engine.massager.massager.Time_period:
            while   self.engine.unstable ==0 and self.engine.massager.massager.time_t < self.engine.massager.massager.Time_period:
                for _ in range(15):
                    self.engine.run()

                # self.engine.massager.step(15)
                self.logger.log(self.engine.get_state())
                # self.renderer.render(self.engine)

        # self.logger.save_logs_to_csv("data/logs/simulation_logs.csv")

    def reset(self):

        print("Resetting simulation...")
        # self.engine.reset()





    def close(self):

        print("Closing simulation...")
        self.recorder.close()



