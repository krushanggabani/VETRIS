
"""
floor.py
Defines the rigid floor boundary.
"""
import taichi as ti

@ti.data_oriented
class Floor:
    def __init__(self, simulator, height=0.0):
        self.sim = simulator
        self.height = height
