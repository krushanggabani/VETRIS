import csv
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import taichi as ti


Number = Union[int, float, np.floating]


class Logger:
    """
    Lightweight simulation logger.

    Expected data payload (dict) per step:
      {
        "time": <float-like>,
        "contact_force": <2D vector-like or scalar>,
        "deformation": <scalar or dict of scalars>,
        # optionally anything else in the future
      }

    Examples for deformation:
      - scalar: 0.0031
      - dict:   {"max_disp": 0.0031, "max_eqv_strain_dev": 0.12}
    """

    def __init__(self, cfg=None, run_dir: Optional[Union[str, Path]] = "logs", run_name: Optional[str] = None):
        self.cfg = cfg
        ts = time.strftime("%Y%m%d-%H%M%S") if run_name is None else str(run_name)
        self.run_dir = Path(run_dir) / ts
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Accumulate rows as plain dicts with normalized scalar values
        self._rows: List[Dict[str, Number]] = []

        # Track which deformation keys we've seen to write consistent CSV columns
        self._deformation_keys: List[str] = []

        # print(f"Logger initialized → {self.run_dir}")

    # -------------------------
    # public API
    # -------------------------
    def log(self, data: Dict[str, Any]) -> None:
        """Append one normalized row from raw state dict."""
        t = self._num(data.get("time"))
        fx, fy, fnorm = self._force_components(data.get("contact_force"))
        deform_items = self._deformation_items(data.get("deformation"))

        row: Dict[str, Number] = {
            "time": t,
            "force_x": fx,
            "force_y": fy,
            "force_norm": fnorm,
        }
        row.update(deform_items)

        # Remember deformation keys order (first-seen order)
        for k in deform_items.keys():
            if k not in self._deformation_keys:
                self._deformation_keys.append(k)

        self._rows.append(row)

    def get_logs(self) -> Dict[str, np.ndarray]:
        """Return columns as numpy arrays (best-effort for present columns)."""
        cols = self._planned_columns()
        out: Dict[str, np.ndarray] = {}
        for c in cols:
            out[c] = np.asarray([r.get(c, np.nan) for r in self._rows], dtype=np.float64)
        return out

    def save_logs_to_csv(self, filename: Optional[Union[str, Path]] = None) -> Path:
        """
        Save CSV with stable columns:
          time, force_x, force_y, force_norm, <deformation keys...>
        """
        if filename is None:
            filename = self.run_dir / "contact_log.csv"
        else:
            filename = Path(filename)

        cols = self._planned_columns()
        with open(filename, "w", newline="") as f:   
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in self._rows:
                # fill missing with NaN
                w.writerow({c: r.get(c, np.nan) for c in cols})
        # print(f"Saved {len(self._rows)} rows → {filename}")
        return filename

    def reset(self) -> None:
        """Clear all accumulated rows."""
        self._rows.clear()
        self._deformation_keys.clear()

    # -------------------------
    # internals
    # -------------------------
    def _planned_columns(self) -> List[str]:
        # Stable base + any seen deformation fields
        base = ["time", "force_x", "force_y", "force_norm"]
        return base + list(self._deformation_keys)

    def _num(self, x: Any) -> float:
        """Best-effort conversion to scalar float (handles Taichi/Numpy/Python)."""
        if hasattr(x, "to_numpy"):
            x = x.to_numpy()
        try:
            arr = np.asarray(x).astype(np.float64)
            if arr.ndim == 0:
                val = float(arr)
            else:
                # If accidentally vector, take first elem
                val = float(arr.reshape(-1)[0])
        except Exception:
            val = float("nan")
        if not np.isfinite(val):
            return float("nan")
        return val

    def _force_components(self, f: Any) -> Tuple[float, float, float]:
        """Return (fx, fy, ||f||). Accepts scalar or vector-like (len>=2)."""
        if hasattr(f, "to_numpy"):
            f = f.to_numpy()
        try:
            arr = np.asarray(f, dtype=np.float64).reshape(-1)
            if arr.size == 0:
                return (float("nan"), float("nan"), float("nan"))
            if arr.size == 1:
                fx = float(arr[0])
                return (fx, 0.0, abs(fx))
            fx = float(arr[0])
            fy = float(arr[1])
            fn = float(np.hypot(fx, fy))
            return (fx, fy, fn)
        except Exception:
            val = self._num(f)
            return (val, 0.0, abs(val) if np.isfinite(val) else float("nan"))

    def _deformation_items(self, d: Any) -> Dict[str, Number]:
        """
        Normalize deformation:
          - if scalar → {"deformation": scalar}
          - if dict   → keep numeric scalars with their keys
        """
        if d is None:
            return {}
        # Dict path (e.g., {"max_disp":..., "max_eqv_strain_dev":...})
        if isinstance(d, dict):
            out: Dict[str, Number] = {}
            for k, v in d.items():
                out[str(k)] = self._num(v)
            return out
        # Scalar path
        return {"deformation": self._num(d)}
