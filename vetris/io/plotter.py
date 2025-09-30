import csv
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

EXPERIMENT_CSV = "data/real/force_indentation.csv"   # two cols: indentation_m, force_N


def load_experiment(csv_path, csv_is_m=False, max_points=None):
    raw = np.loadtxt(csv_path, delimiter=",")
    if raw.ndim != 2 or raw.shape[1] < 2:
        raise ValueError("CSV must have at least two columns: indentation, force")

    indent = raw[:, 0].astype(float)
    force  = raw[:, 1].astype(float)

    if csv_is_m:
        indent = indent * 1e3  # mm -> m



    if max_points is not None and indent.size > max_points:
        idx = np.linspace(0, indent.size - 1, max_points).round().astype(int)
        indent, force = indent[idx], force[idx]
    return indent, force


exp_i, exp_f = load_experiment(EXPERIMENT_CSV, csv_is_m=True, max_points=1580)


class Plotter:
    """
    Load a Logger CSV and visualize how the (strain, force) point moves over time.

    CSV must contain:
      - one of: 'force_norm' | 'contact_force_norm' | 'contact-force-norm' (y-axis)
      - 'max_eqv_strain_contact' (x-axis)
      - optional 'time' column (used only for labels/animation)

    Views:
      - plot()                    : simple scatter of all samples
      - plot_path(step=1, ...)    : line path in time order (with optional arrows)
      - plot_quiver(step=5, ...)  : arrows showing per-step displacement in (strain, force)
      - animate_force_strain(...) : moving dot animation along the path
    """

    FORCE_NORM_ALIASES = ("force_norm", "contact_force_norm", "contact-force-norm")
    STRAIN_KEY = "max_eqv_strain_contact"
    TIME_KEY = "time"

    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        self.rows: List[Dict[str, float]] = []
        self.force_col: str = ""
        self._times: Optional[np.ndarray] = None
        self._load_csv()
        
        

    # ---------------- CSV loading ----------------

    def _get_raw(self):

        return exp_i,exp_f
    
    def _load_csv(self) -> None:
        with open(self.csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            if not cols:
                raise ValueError("CSV has no header / columns.")

            self.force_col = self._find_force_norm_col(cols)
            if self.STRAIN_KEY not in cols:
                raise ValueError(
                    f"Column '{self.STRAIN_KEY}' not found. Available: {cols}"
                )

            rows = []
            times = []
            for r in reader:
                # keep original order (time order) and store as floats (NaN on fail)
                t = self._safe_float(r.get(self.TIME_KEY))
                f_y = self._safe_float(r.get(self.force_col))         # y-axis
                eps_x = self._safe_float(r.get(self.STRAIN_KEY))      # x-axis
                rows.append({self.TIME_KEY: t, self.force_col: f_y, self.STRAIN_KEY: eps_x})
                times.append(t)

            if not rows:
                raise ValueError("No data rows found in CSV.")
            self.rows = rows
            self._times = np.array(times, dtype=float)

    @staticmethod
    def _safe_float(x) -> float:
        try:
            return float(x)
        except Exception:
            return float("nan")

    def _find_force_norm_col(self, cols: List[str]) -> str:
        for name in self.FORCE_NORM_ALIASES:
            if name in cols:
                return name
        raise ValueError(f"None of {self.FORCE_NORM_ALIASES} found in CSV columns: {cols}")

    def _get_xyt(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (x=strain, y=force, t=time) with NaNs removed consistently."""
        x = np.array([r[self.STRAIN_KEY] for r in self.rows], dtype=float)   # strain → x
        y = np.array([r[self.force_col] for r in self.rows], dtype=float)    # force  → y
        t = np.array([r[self.TIME_KEY] for r in self.rows], dtype=float)

        mask = np.isfinite(x) & np.isfinite(y)
        # keep time if present; otherwise synthesize indices 0..n-1
        if not np.any(np.isfinite(t)):
            t = np.arange(len(x), dtype=float)
        x, y, t = x[mask], y[mask], t[mask]
        if x.size == 0:
            raise ValueError("All rows are NaN or missing for the requested columns.")
        return x, y, t

    # ---------------- Simple scatter ----------------

    def plot(self, save_path: Optional[str] = None, show: bool = False) -> Path:

        x, y, _ = self._get_xyt()
        fig = plt.figure(figsize=(12,9))
        ax = fig.add_subplot(111)
        
        ax.plot(x, y,label="Simulated")
        ax.plot(exp_i,exp_f,label="Real")
        ax.set_xlabel(self.STRAIN_KEY)              # x: strain
        ax.set_ylabel("Contact Force Norm")         # y: force
        ax.set_title("Force vs. Strain (y vs. x)")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        ax.legend()
        if save_path is None:
            save_path = self.csv_path.with_name("force_vs_strain_scatter_2.png")
        fig.tight_layout()
        fig.savefig(save_path, dpi=160)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return Path(save_path)

    # ---------------- Path (movement through time) ----------------

    def plot_path(
        self,
        step: int = 1,
        arrows_every: Optional[int] = 10,
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> Path:
        """
        Draw the time-ordered path in (strain, force) space.
        - step: subsample frames (e.g., 2 keeps every other point)
        - arrows_every: put small arrows every K segments (None to disable)
        """
        x, y, t = self._get_xyt()
        idx = np.arange(0, len(x), max(1, step))
        x, y, t = x[idx], y[idx], t[idx]

        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111)
        ax.plot(x, y, linewidth=1.2)       # path
        ax.scatter(x, y, s=10)             # nodes
        ax.scatter([x[0]], [y[0]], s=40, marker='o', label="start")   # start
        ax.scatter([x[-1]], [y[-1]], s=40, marker='s', label="end")   # end

        if arrows_every is not None and arrows_every > 0 and len(x) > 1:
            k = max(1, arrows_every)
            i0 = np.arange(0, len(x) - 1, k)
            dx = x[i0 + 1] - x[i0]
            dy = y[i0 + 1] - y[i0]
            ax.quiver(
                x[i0], y[i0], dx, dy,
                angles="xy", scale_units="xy", scale=1.0, width=0.002
            )

        ax.set_xlabel(self.STRAIN_KEY)          # x: strain
        ax.set_ylabel("Contact Force Norm")     # y: force
        ax.set_title("Time path in force–strain space (y vs. x)")
        ax.grid(True, linestyle="--", linewidth=0.5)
        ax.legend(loc="best")

        if save_path is None:
            save_path = self.csv_path.with_name("force_vs_strain_path.png")
        fig.tight_layout()
        fig.savefig(save_path, dpi=170)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return Path(save_path)

    def plot_quiver(
        self,
        step: int = 5,
        scale: float = 1.0,
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> Path:
        """
        Quiver of per-step motion (Δstrain, Δforce). Good for seeing local dynamics.
        - step: subsample steps to reduce clutter
        - scale: 1.0 means raw Δ; increase to shrink arrows (matplotlib uses 1/scale)
        """
        x, y, _ = self._get_xyt()
        if len(x) < 2:
            raise ValueError("Need at least two samples for quiver.")

        x0 = x[:-1:step]
        y0 = y[:-1:step]
        dx = (x[1:] - x[:-1])[::step]
        dy = (y[1:] - y[:-1])[::step]

        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111)
        ax.quiver(x0, y0, dx, dy, angles="xy", scale_units="xy",
                  scale=(1.0 / max(scale, 1e-9)), width=0.0025)
        ax.set_xlabel(self.STRAIN_KEY)          # x: strain
        ax.set_ylabel("Contact Force Norm")     # y: force
        ax.set_title("Differential motion (quiver) in force–strain space (y vs. x)")
        ax.grid(True, linestyle="--", linewidth=0.5)

        if save_path is None:
            save_path = self.csv_path.with_name("force_vs_strain_quiver.png")
        fig.tight_layout()
        fig.savefig(save_path, dpi=180)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return Path(save_path)

    # ---------------- Animation ----------------

    def animate_force_strain(
        self,
        save_path: str,
        fps: int = 24,
        tail: int = 50,
        dpi: int = 160,
    ) -> Path:
        """
        Animate a moving dot along the (strain, force) path.
        - tail: number of recent samples to keep as a fading trail
        """
        import matplotlib.animation as animation

        x, y, t = self._get_xyt()
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.set_xlabel(self.STRAIN_KEY)          # x: strain
        ax.set_ylabel("Contact Force Norm")     # y: force
        ax.grid(True, linestyle="--", linewidth=0.5)

        # limits with small padding
        pad_x = 0.02 * (np.nanmax(x) - np.nanmin(x) + 1e-9)
        pad_y = 0.02 * (np.nanmax(y) - np.nanmin(y) + 1e-9)
        ax.set_xlim(np.nanmin(x) - pad_x, np.nanmax(x) + pad_x)
        ax.set_ylim(np.nanmin(y) - pad_y, np.nanmax(y) + pad_y)

        (trail_line,) = ax.plot([], [], linewidth=1.5)
        (dot,) = ax.plot([], [], marker="o", markersize=6)

        def init():
            trail_line.set_data([], [])
            dot.set_data([], [])
            ax.set_title("Force–strain trajectory (y vs. x)")
            return trail_line, dot

        def update(i):
            i0 = max(0, i - tail)
            trail_line.set_data(x[i0:i + 1], y[i0:i + 1])
            dot.set_data([x[i]], [y[i]])
            tt = t[i] if np.isfinite(t[i]) else i
            ax.set_title(f"Force–strain trajectory (t={tt:.3f})")
            return trail_line, dot

        ani = animation.FuncAnimation(fig, update, init_func=init,
                                      frames=len(x), interval=1000 / fps, blit=True)
        save_path = Path(save_path)
        # choose writer by extension; requires ffmpeg/imagemagick in your env
        if save_path.suffix.lower() == ".gif":
            ani.save(str(save_path), writer="imagemagick", dpi=dpi, fps=fps)
        else:
            ani.save(str(save_path), writer="ffmpeg", dpi=dpi, fps=fps)
        plt.close(fig)
        return save_path
