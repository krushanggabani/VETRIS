# config.py
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class RecordCfg:
    enabled: bool = True
    output_file: str = "output/simulation.gif"
    fps: int = 30
    resolution: Tuple[int, int] = (1024, 1024)
    palette: str = "muscle"
    show_gui: bool = True
    show_debug: bool = False


@dataclass
class LoggerCfg:
    enabled: bool = True
    log_file: str = "output/simulation.log"
    log_level: str = "info"
    log_to_file: bool = True


@dataclass
class FlagsCfg:
    use_gpu: bool = True
    gpu_id: int = 0
    random_seed: int = 42
    headless: bool = False
    record: RecordCfg = field(default_factory=RecordCfg)
    logger: LoggerCfg = field(default_factory=LoggerCfg)


@dataclass
class VisCfg:
    vis_every: int = 1
    vis_resolution: int = 1024
    vis_palette: str = "muscle"
    vis_window_size: Tuple[int, int] = (1024, 1024)
    vis_window_title: str = "VETRIS Simulation"
    vis_save_screenshot: bool = False
    vis_screenshot_path: str = "output/screenshot.png"
    vis_save_gif: bool = True
    vis_gif_path: str = "output/simulation.gif"
    vis_gif_fps: int = 30
    vis_show_gui: bool = True
    vis_show_debug: bool = False


@dataclass
class MassagerCfg:
    enabled: bool = True
    type: str = "dual_arm"
    control_mode: str = "position"
    initial_positions: Tuple[float, ...] = (0, 0, 0, 0, 0, 0)
    target_positions: Tuple[float, ...] = (0.5, -0.5, 0.5, -0.5, 0.5, -0.5)
    max_velocity: float = 1.0
    max_torque: float = 5.0
    kp: float = 100.0
    kd: float = 20.0


@dataclass
class MPMCfg:
    grid_size: Tuple[int, int] = (128, 128)
    cell_size: float = 0.01
    particle_per_cell: int = 4
    material: str = "muscle"
    density: float = 1000.0

    youngs_modulus: float = 1.0e5
    poisson_ratio: float = 0.3    
    shear_viscosity: float = 5.0
    bulk_viscosity: float = 50.0
    rate_k: float = 0.0
    rate_n: float = 1.0


    friction: float = 0.5
    gravity: Tuple[float, float] = (0.0, -9.81)
    boundary_conditions: str = "solid_wall"
    external_forces: List = field(default_factory=list)
    initial_particles: str = "data/initial_particles.npz"
    external_fields: List = field(default_factory=list)
    collision_objects: List = field(default_factory=list)
    collision_stiffness: float = 1.0e4
    collision_damping: float = 1.0e2
    collision_friction: float = 0.5
    time_integration: str = "implicit"
    implicit_solver_iterations: int = 10
    implicit_solver_tolerance: float = 1.0e-5


@dataclass
class EngineCfg:
    steps: int = 1000
    dt: float = 1.2e-5
    massager: MassagerCfg = field(default_factory=MassagerCfg)
    mpm: MPMCfg = field(default_factory=MPMCfg)


@dataclass
class Config:
    flags: FlagsCfg = field(default_factory=FlagsCfg)
    Vis: VisCfg = field(default_factory=VisCfg)
    engine: EngineCfg = field(default_factory=EngineCfg)


# One global instance
CONFIG = Config()
