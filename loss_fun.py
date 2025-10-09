import os, time, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Literal
from numpy.typing import ArrayLike

# ----------------------------- config ---------------------------------
REF_CSV   = "data/real/loop_2_filtered.csv"      # path to reference CSV
RUNS_DIR  = "data/calibration/exp_27/runs"       # directory with curve_*.csv
START_IDX = 52                                   # start index in sorted file list
N_FILES   = 25                                     # how many files to visualize
PAUSE_SEC = 1000                                    # seconds to show each figure
MAX_PTS   = None                                  # optional downsample cap per file (None to disable)

# loss weights
W1 = 1.0   # branch RMSEs (0.7 load + 0.3 unload)
W2 = 1.0   # d_peak
W3 = 1.0   # d_slope
W4 = 1.0   # d_area
SLOPE_FRAC = 0.2            # fraction for head/tail slope windows
AREA_MODE: Literal["abs","signed"] = "abs"
EXTRAPOLATION: Literal["clamp","linear"] = "linear"
# ----------------------------------------------------------------------

# ======================== utilities & helpers ==========================

def find_column(df, candidates):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None

def normalize_columns(df):
    time_col  = find_column(df, ["# time_s", "time_s", "time", "t"])
    ind_col   = find_column(df, ["indentation_m", "indentation", "indent"])
    force_col = find_column(df, ["force_n", "force_N", "force", "contact_force", "contact_force_n", "F"])
    missing = [name for name, col in [("time", time_col), ("indentation", ind_col), ("force", force_col)] if col is None]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    return time_col, ind_col, force_col

def _to_numeric(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=cols)

def _downsample(df, max_points):
    if max_points is None or len(df) <= max_points:
        return df
    step = max(1, len(df) // max_points)
    return df.iloc[::step].reset_index(drop=True)

def load_and_prepare(csv_path, scale_indent=False):
    df = pd.read_csv(csv_path)
    t, i, f = normalize_columns(df)
    df = df[[t, i, f]].copy()
    df = _to_numeric(df, [t, i, f])

    if scale_indent:
        # Convert mm -> m heuristically (only if values look like mm)
        if df[i].abs().max() > 10:
            df[i] = df[i] * 1e-3
        # Normalize time to [0,1]
        tmax = df[t].abs().max()
        if tmax > 0:
            df[t] = df[t] / tmax
    else:
        # Your original run time scaling (constant)
        denom = (2 * np.pi / 0.4)
        if denom != 0:
            df[t] = df[t] / denom

    df = _downsample(df, MAX_PTS)
    return df, (t, i, f)

def split_cycle(df, time_col, ind_col):
    """Split single-cycle indentation curve at peak indentation."""
    if df.empty:
        return df.copy(), df.copy()
    dfs = df.sort_values(time_col, kind="mergesort").reset_index(drop=True)
    pos_max = int(np.argmax(dfs[ind_col].to_numpy()))
    load_df   = dfs.iloc[:pos_max+1].reset_index(drop=True)
    unload_df = dfs.iloc[pos_max:].reset_index(drop=True)
    return load_df, unload_df

ExtrapMode = Literal["clamp","linear"]
AreaMode   = Literal["abs","signed"]

def _interp_with_extrap(xq: ArrayLike, x: ArrayLike, y: ArrayLike, mode: ExtrapMode) -> np.ndarray:
    x  = np.asarray(x, float); y = np.asarray(y, float); xq = np.asarray(xq, float)
    xu, idx = np.unique(x, return_index=True); yu = y[idx]
    if xu.size == 1:
        return np.full_like(xq, yu[0], dtype=float)
    yi = np.interp(xq, xu, yu)
    if mode == "clamp":
        yi[xq < xu[0]]  = yu[0]
        yi[xq > xu[-1]] = yu[-1]
        return yi
    # linear extrapolation using end slopes
    mL = (yu[1]  - yu[0])  / (xu[1]  - xu[0])
    mR = (yu[-1] - yu[-2]) / (xu[-1] - xu[-2])
    left  = xq < xu[0]; right = xq > xu[-1]
    yi[left]  = yu[0]  + mL * (xq[left]  - xu[0])
    yi[right] = yu[-1] + mR * (xq[right] - xu[-1])
    return yi

def _seg_rmse(exp_x, exp_y, sim_x, sim_y, mode: Literal["overlap","union"]="overlap",
              extrapolation: ExtrapMode="linear") -> Tuple[float, int, np.ndarray, np.ndarray, np.ndarray]:
    exp_x = np.asarray(exp_x, float); exp_y = np.asarray(exp_y, float)
    sim_x = np.asarray(sim_x, float); sim_y = np.asarray(sim_y, float)
    if exp_x.size == 0 or sim_x.size == 0:
        return float("inf"), 0, np.array([]), np.array([]), np.array([])
    if mode == "overlap":
        L = max(np.min(exp_x), np.min(sim_x))
        R = min(np.max(exp_x), np.max(sim_x))
        if not np.isfinite(L) or not np.isfinite(R) or R <= L:
            return float("inf"), 0, np.array([]), np.array([]), np.array([])
        mask = (exp_x >= L) & (exp_x <= R)
        xq = exp_x[mask]
        if xq.size < 2:
            return float("inf"), 0, np.array([]), np.array([]), np.array([])
        exp_on_xq = exp_y[mask]
        sim_on_xq = _interp_with_extrap(xq, sim_x, sim_y, extrapolation)
    else:
        xq = np.unique(np.concatenate([exp_x, sim_x]))
        exp_on_xq = _interp_with_extrap(xq, exp_x, exp_y, "linear")
        sim_on_xq = _interp_with_extrap(xq, sim_x, sim_y, extrapolation)
    diff = sim_on_xq - exp_on_xq
    rmse = float(np.sqrt(np.mean(diff * diff)))
    return rmse, int(xq.size), xq, sim_on_xq, exp_on_xq

def _trapz(y: ArrayLike, x: ArrayLike, mode: AreaMode) -> float:
    y = np.asarray(y, float); x = np.asarray(x, float)
    if y.size < 2:
        return 0.0
    dx = np.diff(x)
    if mode == "abs":
        dx = np.abs(dx)
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * dx))

def _cum_abs_progress(x: ArrayLike) -> np.ndarray:
    x = np.asarray(x, float)
    if x.size == 0:
        return x.copy()
    return np.cumsum(np.abs(np.diff(x, prepend=x[0])))

def _robust_segment_slope(x: ArrayLike, y: ArrayLike, s: ArrayLike,
                          segment: Literal["head","tail"], frac: float = 0.2) -> Tuple[float, np.ndarray]:
    x = np.asarray(x, float); y = np.asarray(y, float); s = np.asarray(s, float)
    if x.size < 2:
        return float("nan"), np.zeros_like(x, dtype=bool)
    smax = s[-1] if s[-1] != 0 else 1.0
    s_norm = s / smax
    m = (s_norm <= frac) if segment == "head" else (s_norm >= (1.0 - frac))
    # fallback to 2-point segment if needed
    if np.count_nonzero(m) < 2:
        m = np.zeros_like(x, dtype=bool)
        if segment == "head":
            m[:2] = True
        else:
            m[-2:] = True
    xs, ys = x[m], y[m]
    if np.allclose(xs.max(), xs.min()):
        return 0.0, m
    p = np.polyfit(xs, ys, 1)
    return float(p[0]), m

# ====================== loss + visualization ===========================

def compute_loss_debug(exp_i, exp_f, sim_i, sim_f,
                       *,
                       slope_frac=SLOPE_FRAC,
                       area_mode=AREA_MODE,
                       extrapolation=EXTRAPOLATION,
                       w1=W1, w2=W2, w3=W3, w4=W4):
    exp_i = np.asarray(exp_i, float); exp_f = np.asarray(exp_f, float)
    sim_i = np.asarray(sim_i, float); sim_f = np.asarray(sim_f, float)

    # Split branches
    (expL_x, expL_y), (expU_x, expU_y) = _split_loading_unloading(exp_i, exp_f)
    (simL_x, simL_y), (simU_x, simU_y) = _split_loading_unloading(sim_i, sim_f)

    # RMSE per branch (overlap)
    rmse_load, nL, xL, sim_on_L, exp_on_L = _seg_rmse(expL_x, expL_y, simL_x, simL_y,
                                                       mode="overlap", extrapolation=extrapolation)
    rmse_unld, nU, xU, sim_on_U, exp_on_U = _seg_rmse(expU_x, expU_y, simU_x, simU_y,
                                                       mode="overlap", extrapolation=extrapolation)
    rmse_all = (np.sqrt(((rmse_load**2)*nL + (rmse_unld**2)*nU) / max(1, (nL+nU)))
                if (nL+nU) else float("inf"))

    # Progress-space alignment for peak/slope/area
    s_exp = _cum_abs_progress(exp_i)
    s_sim = _cum_abs_progress(sim_i)
    sim_f_on_exp = _interp_with_extrap(s_exp, s_sim, sim_f, extrapolation)

    # Peaks/mins
    d_peak = 0.5 * (abs(np.max(sim_f_on_exp) - np.max(exp_f)) +
                    abs(np.min(sim_f_on_exp) - np.min(exp_f)))

    # Slopes (head/tail) using exp_i as x and s_exp for masks; sim aligned onto s_exp
    slope_exp_head, m_head = _robust_segment_slope(exp_i, exp_f, s_exp, segment="head", frac=slope_frac)
    slope_exp_tail, m_tail = _robust_segment_slope(exp_i, exp_f, s_exp, segment="tail", frac=slope_frac)
    slope_sim_head, _      = _robust_segment_slope(exp_i, sim_f_on_exp, s_exp, segment="head", frac=slope_frac)
    slope_sim_tail, _      = _robust_segment_slope(exp_i, sim_f_on_exp, s_exp, segment="tail", frac=slope_frac)
    d_slope = 0.5 * (abs(slope_sim_head - slope_exp_head) + abs(slope_sim_tail - slope_exp_tail))

    # Areas
    area_exp = _trapz(exp_f,        exp_i, area_mode)
    area_sim = _trapz(sim_f_on_exp, exp_i, area_mode)
    d_area   = abs(area_sim - area_exp)

    # Final loss
    L = (w1 * (0.7 * rmse_load + 0.3 * rmse_unld)
         + w2 * d_peak
         + w3 * d_slope
         + w4 * d_area)

    debug = dict(
        rmse_load=rmse_load, nL=nL, xL=xL, sim_on_L=sim_on_L, exp_on_L=exp_on_L,
        rmse_unld=rmse_unld, nU=nU, xU=xU, sim_on_U=sim_on_U, exp_on_U=exp_on_U,
        rmse_all=rmse_all,
        s_exp=s_exp, s_sim=s_sim, sim_f_on_exp=sim_f_on_exp,
        slope_exp_head=slope_exp_head, slope_exp_tail=slope_exp_tail,
        slope_sim_head=slope_sim_head, slope_sim_tail=slope_sim_tail,
        m_head=m_head, m_tail=m_tail,
        d_peak=d_peak, d_slope=d_slope,
        area_exp=area_exp, area_sim=area_sim, d_area=d_area
    )
    comps = {
        "rmse": rmse_all,
        "rmse_load": rmse_load,
        "rmse_unload": rmse_unld,
        "d_peak": d_peak,
        "d_slope": d_slope,
        "d_area": d_area,
        "n_load": nL,
        "n_unload": nU,
        "slope_exp_head": slope_exp_head,
        "slope_exp_tail": slope_exp_tail,
        "slope_sim_head": slope_sim_head,
        "slope_sim_tail": slope_sim_tail,
        "area_exp": area_exp,
        "area_sim": area_sim,
    }
    return float(L), comps, debug

def _split_loading_unloading(x: ArrayLike, y: ArrayLike):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.size == 0:
        z = np.array([], float)
        return (z, z), (z, z)
    pos_max = int(np.argmax(x))
    return (x[:pos_max+1], y[:pos_max+1]), (x[pos_max:], y[pos_max:])

# ============================ plotting =================================

def visualize_loss(exp_i, exp_f, sim_i, sim_f, title):
    L, comps, dbg = compute_loss_debug(exp_i, exp_f, sim_i, sim_f)

    # Build figure (6 subplots) to show each component clearly
    plt.ion()
    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    fig.suptitle(
        f"{title}\n"
        f"L = {L:.4g}  |  "
        f"RMSE(load)={comps['rmse_load']:.4g}, RMSE(unload)={comps['rmse_unload']:.4g}, "
        f"d_peak={comps['d_peak']:.4g}, d_slope={comps['d_slope']:.4g}, d_area={comps['d_area']:.4g}",
        fontsize=12, fontweight='bold'
    )

    # A. Indentation–Force with hysteresis split
    (expL_x, expL_y), (expU_x, expU_y) = _split_loading_unloading(exp_i, exp_f)
    (simL_x, simL_y), (simU_x, simU_y) = _split_loading_unloading(sim_i, sim_f)
    ax = axes[0,0]
    ax.plot(expL_x, expL_y, label="exp: loading", linestyle="--")
    ax.plot(expU_x, expU_y, label="exp: unloading", linestyle=":")
    ax.plot(simL_x, simL_y, label="sim: loading", linestyle="-")
    ax.plot(simU_x, simU_y, label="sim: unloading", linestyle="-.")
    # mark peaks/mins used for d_peak (in progress space we compared forces; mark on exp curve as reference)
    ax.set_xlabel("Indentation (m)"); ax.set_ylabel("Force (N)")
    ax.set_title("Hysteresis split: Indentation vs Force")
    ax.grid(True, alpha=0.3); ax.legend(loc="best")

    # B. RMSE (loading) — show overlap points and vertical errors
    ax = axes[0,1]
    xq, e, s = dbg["xL"], dbg["exp_on_L"], dbg["sim_on_L"]
    if xq.size >= 2:
        ax.plot(xq, e, "o-", label="exp@overlap")
        ax.plot(xq, s, "x--", label="sim@overlap")
        for xi, yi, yj in zip(xq, e, s):
            ax.vlines(xi, min(yi, yj), max(yi, yj), alpha=0.25)
        ax.text(0.02, 0.95, f"RMSE_load = {comps['rmse_load']:.4g}\nN={dbg['nL']}",
                transform=ax.transAxes, va="top", ha="left",
                bbox=dict(boxstyle="round", alpha=0.1))
    else:
        ax.text(0.5, 0.5, "Insufficient overlap", ha="center", va="center")
    ax.set_xlabel("Indentation (m)"); ax.set_ylabel("Force (N)")
    ax.set_title("Loading RMSE (overlap grid)"); ax.grid(True, alpha=0.3); ax.legend(loc="best")

    # C. RMSE (unloading) — show overlap points and vertical errors
    ax = axes[1,0]
    xq, e, s = dbg["xU"], dbg["exp_on_U"], dbg["sim_on_U"]
    if xq.size >= 2:
        ax.plot(xq, e, "o-", label="exp@overlap")
        ax.plot(xq, s, "x--", label="sim@overlap")
        for xi, yi, yj in zip(xq, e, s):
            ax.vlines(xi, min(yi, yj), max(yi, yj), alpha=0.25)
        ax.text(0.02, 0.95, f"RMSE_unload = {comps['rmse_unload']:.4g}\nN={dbg['nU']}",
                transform=ax.transAxes, va="top", ha="left",
                bbox=dict(boxstyle="round", alpha=0.1))
    else:
        ax.text(0.5, 0.5, "Insufficient overlap", ha="center", va="center")
    ax.set_xlabel("Indentation (m)"); ax.set_ylabel("Force (N)")
    ax.set_title("Unloading RMSE (overlap grid)"); ax.grid(True, alpha=0.3); ax.legend(loc="best")

    # D. Area difference (progress-aligned) — fill between curves
    ax = axes[1,1]
    s_exp = dbg["s_exp"]; sim_f_on_exp = dbg["sim_f_on_exp"]
    ax.plot(exp_i, exp_f, label="exp F(i)")
    ax.plot(exp_i, sim_f_on_exp, label="sim F(i) on exp grid", linestyle="--")
    # fill absolute area difference
    ax.fill_between(exp_i, exp_f, sim_f_on_exp, alpha=0.2)
    ax.text(0.02, 0.95, f"d_area = {dbg['d_area']:.4g}", transform=ax.transAxes,
            va="top", ha="left", bbox=dict(boxstyle="round", alpha=0.1))
    ax.set_xlabel("Indentation (m)"); ax.set_ylabel("Force (N)")
    ax.set_title("Area difference under F(i)"); ax.grid(True, alpha=0.3); ax.legend(loc="best")

    # E. Slopes — head window
    ax = axes[2,0]
    m_head = dbg["m_head"]
    if np.count_nonzero(m_head) >= 2:
        xi = exp_i[m_head]; yi_e = exp_f[m_head]; yi_s = dbg["sim_f_on_exp"][m_head]
        # fit lines for viz
        pe = np.polyfit(xi, yi_e, 1); ps = np.polyfit(xi, yi_s, 1)
        xx = np.linspace(xi.min(), xi.max(), 50)
        ax.plot(exp_i, exp_f, alpha=0.25, label="exp (context)")
        ax.plot(xx, np.polyval(pe, xx), label=f"exp head slope={dbg['slope_exp_head']:.3g}")
        ax.plot(xx, np.polyval(ps, xx), linestyle="--", label=f"sim head slope={dbg['slope_sim_head']:.3g}")
        ax.scatter(xi, yi_e, s=10)
        ax.scatter(xi, yi_s, s=10)
    else:
        ax.text(0.5, 0.5, "Head segment too small", ha="center", va="center")
    ax.set_xlabel("Indentation (m)"); ax.set_ylabel("Force (N)")
    ax.set_title("Head slope window (progress-based)"); ax.grid(True, alpha=0.3); ax.legend(loc="best")

    # F. Slopes — tail window + peak markers for d_peak
    ax = axes[2,1]
    m_tail = dbg["m_tail"]
    if np.count_nonzero(m_tail) >= 2:
        xi = exp_i[m_tail]; yi_e = exp_f[m_tail]; yi_s = dbg["sim_f_on_exp"][m_tail]
        pe = np.polyfit(xi, yi_e, 1); ps = np.polyfit(xi, yi_s, 1)
        xx = np.linspace(xi.min(), xi.max(), 50)
        ax.plot(exp_i, exp_f, alpha=0.25, label="exp (context)")
        ax.plot(xx, np.polyval(pe, xx), label=f"exp tail slope={dbg['slope_exp_tail']:.3g}")
        ax.plot(xx, np.polyval(ps, xx), linestyle="--", label=f"sim tail slope={dbg['slope_sim_tail']:.3g}")
        ax.scatter(xi, yi_e, s=10)
        ax.scatter(xi, yi_s, s=10)
    else:
        ax.text(0.5, 0.5, "Tail segment too small", ha="center", va="center")
    # peak/min markers (using progress-aligned sim)
    emax, emin = float(np.max(exp_f)), float(np.min(exp_f))
    smax, smin = float(np.max(dbg["sim_f_on_exp"])), float(np.min(dbg["sim_f_on_exp"]))
    ax.axhline(emax, color="C0", alpha=0.2); ax.axhline(emin, color="C0", alpha=0.2)
    ax.axhline(smax, color="C1", alpha=0.2, linestyle="--"); ax.axhline(smin, color="C1", alpha=0.2, linestyle="--")
    ax.text(0.02, 0.95, f"d_peak = {dbg['d_peak']:.4g}", transform=ax.transAxes,
            va="top", ha="left", bbox=dict(boxstyle="round", alpha=0.1))
    ax.set_xlabel("Indentation (m)"); ax.set_ylabel("Force (N)")
    ax.set_title("Tail slope window + peak/min for d_peak"); ax.grid(True, alpha=0.3); ax.legend(loc="best")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show(block=False)
    print(f"Showing: {title} — waiting {PAUSE_SEC} seconds...")
    plt.pause(PAUSE_SEC)
    plt.close(fig)

# ================================ main =================================

def main():
    # Load reference once
    try:
        ref_df, ref_cols = load_and_prepare(REF_CSV, scale_indent=True)
    except Exception as e:
        print(f"❌ Could not load reference CSV: {REF_CSV}\n   Error: {e}")
        return

    t_ref, i_ref, f_ref = ref_cols

    # Collect run files
    pattern = os.path.join(RUNS_DIR, "curve_*.csv")
    all_files = sorted(glob.glob(pattern))
    files = all_files[START_IDX: START_IDX + N_FILES]

    if not files:
        print(f"No files found matching {pattern} in requested range.")
        return

    print(f"Found {len(files)} files (starting at index {START_IDX}).\n")

    for idx, fpath in enumerate(files, start=1):
        try:
            run_df, run_cols = load_and_prepare(fpath, scale_indent=False)
            t_r, i_r, f_r = run_cols

            # Use indentation–force pairs (exp vs sim) for loss/visualization
            exp_i, exp_f = ref_df[i_ref].to_numpy(), ref_df[f_ref].to_numpy()
            sim_i, sim_f = run_df[i_r].to_numpy(), run_df[f_r].to_numpy()

            stem  = os.path.splitext(os.path.basename(fpath))[0]
            title = f"[{idx}/{len(files)}] {stem}"
            visualize_loss(exp_i, exp_f, sim_i, sim_f, title)

        except Exception as e:
            print(f"⚠️  Skipped {fpath}: {e}")

    print("\nDone. All figures displayed and closed automatically.")

if __name__ == "__main__":
    main()
