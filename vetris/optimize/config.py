from dataclasses import dataclass
from typing import Optional

@dataclass
class CalibConfig:
    experiment_csv: str = "data/real/loop_1_filtered.csv"
    csv_is_mm: bool = True
    massager_type: str = "straight"     # "straight" | "dual_arm"
    override_time_period: Optional[float] = None

    # Grid / CFL
    rho: float = 1.0
    n_grid_dt: int = 128
    cfl_number: float = 0.38
    dt_min: float = 5e-7
    dt_max: float = 5e-3

    # Modes
    coarse_trials: int = 100
    coarse_sigma: float = 0.6
    do_coarse: bool = True
    do_fine: bool = True
    exp_max_points: Optional[int] = 1580

    # Bounds & init
    from dataclasses import field
    from typing import List, Tuple

    bounds: List[Tuple[float, float]] = field(default_factory=lambda: [
        (1e3, 5e6),   # E
        (0.01, 0.49), # nu
        (1.0, 1e4),   # shear viscosity
        (1.0, 1e5),   # bulk viscosity
        (0.0, 5.0),   # rate_k
        (0.2, 2.5),   # rate_n
    ])
    x0: List[float] = field(default_factory=lambda: [1.72e4, 0.204, 2.61e1, 2.532133e+00, 0.7173, 1.52])

    # x0: List[float] = field(default_factory=lambda: [1.72e4, 0.204, 2.61e1, 2.532133e+00])

    weights: List[float] = field(default_factory=lambda: [50.0, 0.5, 0.01, 100.0])
    # Output root
    file_location: str = "data/calibration"
    exp_subdir: str = "exp_2"


