import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Literal, Optional

# NEW: matplotlib is optional; import lazily in plotting helper
def _ensure_mpl():
    import importlib
    plt = importlib.import_module("matplotlib.pyplot")
    return plt

ArrayLike = np.ndarray
ExtrapMode = Literal["clamp", "linear"]
AreaMode   = Literal["abs", "signed"]
MethodMode = Literal["rmse_overlapped", "rmse", "weighted_sum"]


# ---------- utilities ----------
def _trapz(y: ArrayLike, x: ArrayLike, mode: AreaMode) -> float:
    y = np.asarray(y, float); x = np.asarray(x, float)
    if y.size < 2: return 0.0
    dx = np.diff(x)
    if mode == "abs": dx = np.abs(dx)
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * dx))

def _cum_abs_progress(x: ArrayLike) -> ArrayLike:
    x = np.asarray(x, float)
    if x.size == 0: return x
    return np.cumsum(np.abs(np.diff(x, prepend=x[0])))

def _interp_with_extrap(xq: ArrayLike, x: ArrayLike, y: ArrayLike, mode: ExtrapMode) -> ArrayLike:
    x  = np.asarray(x, float); y = np.asarray(y, float); xq = np.asarray(xq, float)
    xu, idx = np.unique(x, return_index=True); yu = y[idx]
    if xu.size == 1: return np.full_like(xq, yu[0], dtype=float)
    yi = np.interp(xq, xu, yu)
    if mode == "clamp":
        yi[xq < xu[0]]  = yu[0]
        yi[xq > xu[-1]] = yu[-1]
        return yi
    mL = (yu[1]  - yu[0])  / (xu[1]  - xu[0])
    mR = (yu[-1] - yu[-2]) / (xu[-1] - xu[-2])
    left  = xq < xu[0]; right = xq > xu[-1]
    yi[left]  = yu[0]  + mL * (xq[left]  - xu[0])
    yi[right] = yu[-1] + mR * (xq[right] - xu[-1])
    return yi

def _robust_segment_slope(
    x: ArrayLike, y: ArrayLike, s: ArrayLike, *,
    segment: Literal["head", "tail"], frac: float = 0.2, eps: float = 1e-9
) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float); s = np.asarray(s, float)
    if x.size < 2: return 0.0
    smax = s[-1] if s.size else 0.0
    if smax <= 0.0:
        a, _ = np.polyfit(x, y, 1); return float(a)
    if segment == "head":
        mp = (s[:-1] <= frac*smax) & (s[1:] <= frac*smax); ms = (s <= frac*smax)
    else:
        mp = (s[:-1] >= (1-frac)*smax) & (s[1:] >= (1-frac)*smax); ms = (s >= (1-frac)*smax)
    dx = x[1:] - x[:-1]; dy = y[1:] - y[:-1]
    good = mp & (np.abs(dx) > eps)
    if np.any(good): return float(np.median(dy[good]/dx[good]))
    if np.count_nonzero(ms) >= 2:
        a, _ = np.polyfit(x[ms], y[ms], 1); return float(a)
    a, _ = np.polyfit(x, y, 1); return float(a)

def _split_loading_unloading(x: ArrayLike, y: ArrayLike, eps: float = 1e-12):
    x = np.asarray(x, float); y = np.asarray(y, float)
    
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(x, prepend=y[0])
    m_load = dy > eps     # increasing indentation
    m_unld = dy < -eps    # decreasing indentation
    return (x[m_load], y[m_load]), (x[m_unld], y[m_unld])

def _seg_rmse(
    exp_x: ArrayLike, exp_y: ArrayLike,
    sim_x: ArrayLike, sim_y: ArrayLike,
    *, mode: Literal["overlap", "no_overlap"], extrapolation: ExtrapMode
) -> Tuple[float, int, ArrayLike, ArrayLike, ArrayLike]:
    """Returns rmse, n, x_eval, sim_on_eval, exp_on_eval (exp_y subset) for plotting."""
    exp_x = np.asarray(exp_x, float); exp_y = np.asarray(exp_y, float)
    sim_x = np.asarray(sim_x, float); sim_y = np.asarray(sim_y, float)
    if exp_x.size < 2 or sim_x.size < 2:
        return float("inf"), 0, exp_x[:0], exp_x[:0], exp_x[:0]
    if mode == "overlap":
        lo = max(exp_x.min(), sim_x.min()); hi = min(exp_x.max(), sim_x.max())
        m = (exp_x >= lo) & (exp_x <= hi)
        if not np.any(m): return float("inf"), 0, exp_x[:0], exp_x[:0], exp_x[:0]
        x_eval = exp_x[m]
        sim_on = _interp_with_extrap(x_eval, sim_x, sim_y, "linear")
        exp_on = exp_y[m]
    else:
        x_eval = exp_x
        sim_on = _interp_with_extrap(x_eval, sim_x, sim_y, extrapolation)
        exp_on = exp_y
    resid = sim_on - exp_on
    rmse = float(np.sqrt(np.mean(resid**2)))
    return rmse, int(x_eval.size), x_eval, sim_on, exp_on


# ---------- plotting helper (centralized) ----------
def _plot_stage(
    *, title: str,
    exp_i: ArrayLike, exp_f: ArrayLike,
    sim_i: ArrayLike, sim_f: ArrayLike,
    expL=None, expU=None, simL=None, simU=None,
    seg_name: Optional[str]=None, x_eval=None, sim_on=None, exp_on=None,
    show_progress=False, s_exp=None, s_sim=None, sim_f_on_exp=None,
    show_area=False, area_mode: AreaMode="abs"
):
    plt = _ensure_mpl()
    ncols = 2 + int(seg_name is not None) + int(show_progress) + int(show_area)
    fig, axes = plt.subplots(1, ncols, figsize=(4.5*ncols, 4.2), constrained_layout=True)
    if ncols == 1: axes = [axes]

    # 1) raw curves + split masks
    ax = axes[0]
    ax.plot(exp_i, exp_f, label="EXP (raw)", linewidth=1.5)
    ax.plot(sim_i, sim_f, label="SIM (raw)", linewidth=1.5, linestyle="--")
    if expL is not None:
        ax.scatter(expL[0], expL[1], s=12, label="EXP loading")
    if expU is not None:
        ax.scatter(expU[0], expU[1], s=12, label="EXP unloading")
    if simL is not None:
        ax.scatter(simL[0], simL[1], s=12, label="SIM loading", marker="x")
    if simU is not None:
        ax.scatter(simU[0], simU[1], s=12, label="SIM unloading", marker="x")
    ax.set_xlabel("indentation (x)")
    ax.set_ylabel("force (y)")
    ax.set_title("Raw + split")
    ax.legend(fontsize=8)

    # 2) segment window (if provided)
    idx = 1
    if seg_name is not None:
        ax2 = axes[idx]; idx += 1
        ax2.plot(exp_i, exp_f, linewidth=1.0, alpha=0.4)
        ax2.plot(sim_i, sim_f, linewidth=1.0, alpha=0.4, linestyle="--")
        if x_eval is not None and exp_on is not None and sim_on is not None and x_eval.size:
            ax2.plot(x_eval, exp_on, label=f"{seg_name}: EXP on eval", linewidth=1.8)
            ax2.plot(x_eval, sim_on, label=f"{seg_name}: SIM→eval", linewidth=1.8, linestyle="--")
            resid = sim_on - exp_on
            ax2.set_title(f"{seg_name}: interp & residual (RMSE={np.sqrt(np.mean(resid**2)):.3g})")
        else:
            ax2.set_title(f"{seg_name}: no overlap / no data")
        ax2.set_xlabel("indentation on segment")
        ax2.set_ylabel("force")
        ax2.legend(fontsize=8)

    # 3) progress-space alignment (for weighted_sum)
    if show_progress and (s_exp is not None) and (s_sim is not None) and (sim_f_on_exp is not None):
        ax3 = axes[idx]; idx += 1
        ax3.plot(s_exp, exp_f, label="EXP: f(s)")
        ax3.plot(s_exp, sim_f_on_exp, label="SIM→EXP: f(s)")
        ax3.set_xlabel("progress s (cum |Δx|)")
        ax3.set_ylabel("force")
        ax3.set_title("Progress-space alignment")
        ax3.legend(fontsize=8)

    # 4) area visualization (optional)
    if show_area:
        ax4 = axes[idx]
        ax4.plot(exp_i, exp_f, label="EXP", linewidth=1.5)
        ax4.plot(exp_i, sim_f_on_exp if sim_f_on_exp is not None else np.nan, label="SIM→EXP", linestyle="--")
        # simple fill to show difference in loop energy along path (visual aid only)
        if sim_f_on_exp is not None and exp_i.size == sim_f_on_exp.size:
            ax4.fill_between(exp_i, exp_f, sim_f_on_exp, alpha=0.2)
        ax4.set_title(f"Areas (mode={area_mode})")
        ax4.set_xlabel("indentation (x)")
        ax4.set_ylabel("force (y)")
        ax4.legend(fontsize=8)

    fig.suptitle(title, fontsize=12)
    plt.show()


# =================== MAIN CLASS WITH PLOTTING ===================
@dataclass
class HysteresisLoss:
    """
    Visual debug: set debug_plot=True to see step-by-step figures.
    """
    weights: Tuple[float, float, float, float] = (1.0, 0.5, 1e-3, 50.0)
    extrapolation: ExtrapMode = "clamp"
    slope_frac: float = 0.2
    area_mode: AreaMode = "abs"
    method: MethodMode = "weighted_sum"
    min_points: int = 25
    debug_plot: bool = False  # <--- NEW

    def __call__(self, exp_i, exp_f, sim_i, sim_f):
        if self.method == "rmse_overlapped":
            return self.rmse_overlapped(exp_i, exp_f, sim_i, sim_f)
        elif self.method == "rmse":
            return self.rmse_no_overlap(exp_i, exp_f, sim_i, sim_f)
        elif self.method == "weighted_sum":
            return self.weighted_sum(exp_i, exp_f, sim_i, sim_f)
        raise ValueError("Unknown method")

    # ---------- RMSE (no-overlap) with plots ----------
    def rmse_no_overlap(self, exp_i, exp_f, sim_i, sim_f):
        exp_i = np.asarray(exp_i, float); exp_f = np.asarray(exp_f, float)
        sim_i = np.asarray(sim_i, float); sim_f = np.asarray(sim_f, float)
        if exp_i.size < self.min_points or sim_i.size < self.min_points:
            comps = {"rmse": float("inf"), "rmse_load": float("inf"), "rmse_unload": float("inf")}
            return float("inf"), comps, 0

        (expL_x, expL_y), (expU_x, expU_y) = _split_loading_unloading(exp_i, exp_f)
        (simL_x, simL_y), (simU_x, simU_y) = _split_loading_unloading(sim_i, sim_f)


        rmse_load, nL, xL, sim_on_L, exp_on_L = _seg_rmse(
            expL_x, expL_y, simL_x, simL_y, mode="no_overlap", extrapolation=self.extrapolation
        )
        rmse_unld, nU, xU, sim_on_U, exp_on_U = _seg_rmse(
            expU_x, expU_y, simU_x, simU_y, mode="no_overlap", extrapolation=self.extrapolation
        )

        if self.debug_plot:
            _plot_stage(
                title="RMSE (no-overlap): loading segment",
                exp_i=exp_i, exp_f=exp_f, sim_i=sim_i, sim_f=sim_f,
                expL=(expL_x, expL_y), expU=(expU_x, expU_y), simL=(simL_x, simL_y), simU=(simU_x, simU_y),
                seg_name=f"Loading (n={nL}, rmse={rmse_load:.3g})", x_eval=xL, sim_on=sim_on_L, exp_on=exp_on_L
            )
            _plot_stage(
                title="RMSE (no-overlap): unloading segment",
                exp_i=exp_i, exp_f=exp_f, sim_i=sim_i, sim_f=sim_f,
                expL=(expL_x, expL_y), expU=(expU_x, expU_y), simL=(simL_x, simL_y), simU=(simU_x, simU_y),
                seg_name=f"Unloading (n={nU}, rmse={rmse_unld:.3g})", x_eval=xU, sim_on=sim_on_U, exp_on=exp_on_U
            )

        rmse_all = (np.sqrt(((rmse_load**2)*nL + (rmse_unld**2)*nU) / max(1, (nL+nU)))
                    if (nL+nU) else float("inf"))

        comps = {"rmse": rmse_all, "rmse_load": rmse_load, "rmse_unload": rmse_unld,
                 "n_load": nL, "n_unload": nU}
        return rmse_all, comps, int(nL + nU)

    # ---------- RMSE (overlap-only) with plots ----------
    def rmse_overlapped(self, exp_i, exp_f, sim_i, sim_f):
        exp_i = np.asarray(exp_i, float); exp_f = np.asarray(exp_f, float)
        sim_i = np.asarray(sim_i, float); sim_f = np.asarray(sim_f, float)
        if exp_i.size < self.min_points or sim_i.size < self.min_points:
            comps = {"rmse": float("inf"), "rmse_load": float("inf"), "rmse_unload": float("inf")}
            return float("inf"), comps, 0

        (expL_x, expL_y), (expU_x, expU_y) = _split_loading_unloading(exp_i, exp_f)
        (simL_x, simL_y), (simU_x, simU_y) = _split_loading_unloading(sim_i, sim_f)

        rmse_load, nL, xL, sim_on_L, exp_on_L = _seg_rmse(
            expL_x, expL_y, simL_x, simL_y, mode="overlap", extrapolation="linear"
        )
        rmse_unld, nU, xU, sim_on_U, exp_on_U = _seg_rmse(
            expU_x, expU_y, simU_x, simU_y, mode="overlap", extrapolation="linear"
        )

        if self.debug_plot:
            _plot_stage(
                title="RMSE (overlap): loading segment",
                exp_i=exp_i, exp_f=exp_f, sim_i=sim_i, sim_f=sim_f,
                expL=(expL_x, expL_y), expU=(expU_x, expU_y), simL=(simL_x, simL_y), simU=(simU_x, simU_y),
                seg_name=f"Loading (n={nL}, rmse={rmse_load:.3g})", x_eval=xL, sim_on=sim_on_L, exp_on=exp_on_L
            )
            _plot_stage(
                title="RMSE (overlap): unloading segment",
                exp_i=exp_i, exp_f=exp_f, sim_i=sim_i, sim_f=sim_f,
                expL=(expL_x, expL_y), expU=(expU_x, expU_y), simL=(simL_x, simL_y), simU=(simU_x, simU_y),
                seg_name=f"Unloading (n={nU}, rmse={rmse_unld:.3g})", x_eval=xU, sim_on=sim_on_U, exp_on=exp_on_U
            )

        rmse_all = (np.sqrt(((rmse_load**2)*nL + (rmse_unld**2)*nU) / max(1, (nL+nU)))
                    if (nL+nU) else float("inf"))

        comps = {"rmse": rmse_all, "rmse_load": rmse_load, "rmse_unload": rmse_unld,
                 "n_load": nL, "n_unload": nU}
        return rmse_all, comps, int(nL + nU)

    # ---------- Weighted sum with progress/area plots ----------
    def weighted_sum(self, exp_i, exp_f, sim_i, sim_f):
        w1, w2, w3, w4 = self.weights
        exp_i = np.asarray(exp_i, float); exp_f = np.asarray(exp_f, float)
        sim_i = np.asarray(sim_i, float); sim_f = np.asarray(sim_f, float)
        if exp_i.size < self.min_points or sim_i.size < self.min_points:
            comps = {"rmse": float("inf"), "rmse_load": float("inf"), "rmse_unload": float("inf"),
                     "d_peak": float("inf"), "d_slope": float("inf"), "d_area": float("inf")}
            return float("inf"), comps, 0

        # segment RMSEs (use overlap policy here; change to 'no_overlap' if you prefer)
        (expL_x, expL_y), (expU_x, expU_y) = _split_loading_unloading(exp_i, exp_f)
        (simL_x, simL_y), (simU_x, simU_y) = _split_loading_unloading(sim_i, sim_f)


        

        rmse_load, nL, xL, sim_on_L, exp_on_L = _seg_rmse(expL_x, expL_y, simL_x, simL_y, mode="overlap", extrapolation="linear")
        rmse_unld, nU, xU, sim_on_U, exp_on_U = _seg_rmse(expU_x, expU_y, simU_x, simU_y, mode="overlap", extrapolation="linear")
        rmse_all = (np.sqrt(((rmse_load**2)*nL + (rmse_unld**2)*nU) / max(1, (nL+nU)))
                    if (nL+nU) else float("inf"))

        # progress-space alignment for peaks/slopes/area
        s_exp = _cum_abs_progress(exp_i)
        s_sim = _cum_abs_progress(sim_i)
        sim_f_on_exp = _interp_with_extrap(s_exp, s_sim, sim_f, self.extrapolation)

        if self.debug_plot:
            # show both segments quickly, plus progress and area
            _plot_stage(
                title="Weighted sum: loading segment (overlap)",
                exp_i=exp_i, exp_f=exp_f, sim_i=sim_i, sim_f=sim_f,
                expL=(expL_x, expL_y), expU=(expU_x, expU_y), simL=(simL_x, simL_y), simU=(simU_x, simU_y),
                seg_name=f"Loading (n={nL}, rmse={rmse_load:.3g})", x_eval=xL, sim_on=sim_on_L, exp_on=exp_on_L,
                show_progress=True, s_exp=s_exp, s_sim=s_sim, sim_f_on_exp=sim_f_on_exp,
                show_area=True, area_mode=self.area_mode
            )
            _plot_stage(
                title="Weighted sum: unloading segment (overlap)",
                exp_i=exp_i, exp_f=exp_f, sim_i=sim_i, sim_f=sim_f,
                expL=(expL_x, expL_y), expU=(expU_x, expU_y), simL=(simL_x, simL_y), simU=(simU_x, simU_y),
                seg_name=f"Unloading (n={nU}, rmse={rmse_unld:.3g})", x_eval=xU, sim_on=sim_on_U, exp_on=exp_on_U
            )

        # other components
        d_peak = 0.5 * (abs(np.max(sim_f_on_exp) - np.max(exp_f)) +
                        abs(np.min(sim_f_on_exp) - np.min(exp_f)))

        slope_exp_head = _robust_segment_slope(exp_i, exp_f, s_exp, segment="head", frac=self.slope_frac)
        slope_exp_tail = _robust_segment_slope(exp_i, exp_f, s_exp, segment="tail", frac=self.slope_frac)
        slope_sim_head = _robust_segment_slope(exp_i, sim_f_on_exp, s_exp, segment="head", frac=self.slope_frac)
        slope_sim_tail = _robust_segment_slope(exp_i, sim_f_on_exp, s_exp, segment="tail", frac=self.slope_frac)
        d_slope = 0.5 * (abs(slope_sim_head - slope_exp_head) + abs(slope_sim_tail - slope_exp_tail))

        area_exp = _trapz(exp_f,        exp_i, self.area_mode)
        area_sim = _trapz(sim_f_on_exp, exp_i, self.area_mode)
        d_area   = abs(area_sim - area_exp)

        L = w1 * rmse_all + w2 * d_peak + w3 * d_slope + w4 * d_area
        comps: Dict[str, float] = {
            "rmse": rmse_all, "rmse_load": rmse_load, "rmse_unload": rmse_unld,
            "n_load": nL, "n_unload": nU,
            "d_peak": d_peak, "d_slope": d_slope, "d_area": d_area,
            "slope_exp_head": slope_exp_head, "slope_exp_tail": slope_exp_tail,
            "slope_sim_head": slope_sim_head, "slope_sim_tail": slope_sim_tail,
            "area_exp": area_exp, "area_sim": area_sim,
        }
        print(comps)
        return float(L), comps, int(nL + nU)
