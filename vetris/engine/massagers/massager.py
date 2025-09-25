import taichi as ti
import numpy as np


from vetris.engine.massagers.straight import straight_massager
from vetris.engine.massagers.dual_arm import dual_arm_massager


@ti.data_oriented
class massager:
    def __init__(self, cfg):

        self.cfg = cfg.engine.massager
        self.cfg.dt = cfg.engine.dt
        self.type = self.cfg.type
        

        if self.type == "straight":
            self.massager = straight_massager(self.cfg)
            print("Material Testing setup initialized")
        elif self.type == "dual_arm":
            self.massager = dual_arm_massager(self.cfg)
            print("Dual-arm massager initialized")
        else:
            raise ValueError(f"Unknown massager type: {self.type}")

    def initialize(self):
        self.massager.initialize()

    def step(self,n_substeps):
        self.massager.step(n_substeps)

     # unify dt from engine (optional)
    def set_dt(self, dt: float):
        if hasattr(self.massager, "set_dt"):
            self.massager.set_dt(dt)

    # reset contact force prior to grid update
    def zero_contact_force(self):
        if hasattr(self.massager, "zero_contact_force"):
            self.massager.zero_contact_force()

            