import taichi as ti
import numpy as np
from typing import Optional, Tuple, Union
from vetris.vis.palette import tissue_palettes

ArrayLike = Union[np.ndarray, float, int]

# ──────────────────────────────────────────────────────────────────────────────
# Utilities (vectorized, allocation-aware)
# ──────────────────────────────────────────────────────────────────────────────
class RenderUtils:
    @staticmethod
    def to_numpy_safe(x: ArrayLike) -> np.ndarray:
        """Accepts Taichi field/ndarray/list/scalar; returns float32 numpy array."""
        if hasattr(x, "to_numpy"):
            arr = x.to_numpy()
        else:
            arr = np.asarray(x)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    @staticmethod
    def safe_minmax_norm(a: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        # Assumes a already float32 and finite (via to_numpy_safe)
        r = float(np.ptp(a))
        if not np.isfinite(r) or r < eps:
            return np.zeros_like(a, dtype=np.float32)
        amin = float(np.min(a))
        out = (a - amin) / (r + eps)
        return np.clip(out, 0.0, 1.0, out=out)

    @staticmethod
    def pack_rgb_array(rgb_arr: np.ndarray) -> np.ndarray:
        # rgb_arr expected in [0,255] float or int; returns uint32 packed
        a = np.nan_to_num(rgb_arr, nan=0.0, posinf=255.0, neginf=0.0)
        a = np.clip(a, 0, 255).astype(np.uint32, copy=False)
        return (a[:, 0] << 16) + (a[:, 1] << 8) + a[:, 2]

    @staticmethod
    def force_mag(obj: ArrayLike) -> float:
        """Safe Euclidean norm for field/ndarray/list/scalar; returns float."""
        if hasattr(obj, "to_numpy"):
            arr = obj.to_numpy()
        else:
            arr = np.asarray(obj)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.ndim == 0:
            return float(arr)
        return float(np.linalg.norm(arr))

# ──────────────────────────────────────────────────────────────────────────────
# Palette helpers
# ──────────────────────────────────────────────────────────────────────────────
def get_palette(name: str = "muscle"):
    return tissue_palettes.get(name, tissue_palettes["classic"])

def get_tissue_colors(palette) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # store as float32 arrays to enable vector ops without casting every frame
    low  = np.asarray(palette["low"], dtype=np.float32)
    mid  = np.asarray(palette["mid"], dtype=np.float32)
    high = np.asarray(palette["high"], dtype=np.float32)
    return low, mid, high

# Arm draw palette (kept identical)
ARM_BODY    = 0x1F2937
ARM_EDGE    = 0x374151
JOINT_COLOR = 0xFF7A59
ROLLER_EDGE = 0x222222


# ──────────────────────────────────────────────────────────────────────────────
# Renderer (single GUI instance, zero-copy paths where possible)
# ──────────────────────────────────────────────────────────────────────────────
class Renderer:
    def __init__(self, cfg):

        self.cfg = cfg.Vis
        self.massager_tyepe = cfg.engine.massager.type

        self.resolution = getattr(cfg.Vis,"resolution", 1024)
        self.gui = ti.GUI('MPM + Single Passive Arm (Attached & Outward)', res=(self.resolution, self.resolution))

        # palette / tissue colors (vector form for fast lerp)
        self.palette = get_palette(getattr(cfg.Vis,"palette_name", "muscle"))
        self.TISSUE_LOW, self.TISSUE_MID, self.TISSUE_HIGH = get_tissue_colors(self.palette)

        # cached small vectors to avoid per-frame allocations
        self._shadow_offset = np.array([0.0025, -0.0025], dtype=np.float32)
        self._arm_shadow    = np.array([0.003, -0.003], dtype=np.float32)
        self._edge_lit      = np.array([+0.0015, +0.0015], dtype=np.float32)

        # HUD positions (immutable)
        self._hud_pos_time  = (0.02, 0.95)
        self._hud_pos_fL    = (0.02, 0.92)
        self._hud_pos_fR    = (0.02, 0.89)

        

    # ──────────────────────────────────────────────────────────────────────────
    # Public API (same modalities as original)
    # ──────────────────────────────────────────────────────────────────────────
    def render(self, engine):

        particles = engine.x
        massager = engine.massager.massager
        
        
             
        # 1) Particles to numpy (once)
        x_np = RenderUtils.to_numpy_safe(particles)
        # Defensive: ensure shape (N,2)
        x_np = x_np.reshape(-1, 2)

        # 2) Position-only shading (fully vectorized)
        t_pos = RenderUtils.safe_minmax_norm(x_np[:, 1])  # (N,)
        t_lo  = np.clip(2.0 * t_pos, 0.0, 1.0)
        t_hi  = np.clip(2.0 * (t_pos - 0.5), 0.0, 1.0)

        # lerp in RGB space; keep float32
        # base01 = mix(low, mid, t_lo), base = mix(base01, high, t_hi)
        base01 = (1.0 - t_lo[:, None]) * self.TISSUE_LOW + t_lo[:, None] * self.TISSUE_MID
        base   = (1.0 - t_hi[:, None]) * base01        + t_hi[:, None] * self.TISSUE_HIGH

        # vignette-like edge darkening
        centroid = np.mean(x_np, axis=0, keepdims=True)
        edge_w   = RenderUtils.safe_minmax_norm(np.linalg.norm(x_np - centroid, axis=1))
        base    *= (1.0 - 0.15 * edge_w[:, None])

        colors_uint32 = RenderUtils.pack_rgb_array(base)



        # using strain
        # eq_np  = engine.eqv_strain_dev.to_numpy()
        # s = (eq_np - np.min(eq_np)) / (np.ptp(eq_np) + 1e-12)
        # s = s[:, None]
        # base = base + (1.0 - s) * self.TISSUE_LOW + s * self.TISSUE_HIGH
        # colors_uint32 = RenderUtils.pack_rgb_array(base)

        # 3) Tissue particles + shadow
        self.gui.circles(x_np + self._shadow_offset, radius=2.2, color=0x151515)
        self.gui.circles(x_np,                       radius=1.8, color=colors_uint32)



        if self.massager_tyepe == "dual_arm":
            # 4) Contact halos (kept behavior, but with clean "None means skip")
            ee_l, cf_l = self._end_eff_and_force(massager.roller_center_left, massager.contact_force_vec_l)
            ee_r, cf_r = self._end_eff_and_force(massager.roller_center_right, massager.contact_force_vec_r)
            if ee_l is not None:
                halo_r = int(np.clip(6 + 0.004 * cf_l, 6, 22))
                self.gui.circle(ee_l, radius=halo_r, color=0xFFFFFF)
            if ee_r is not None:
                halo_r = int(np.clip(6 + 0.004 * cf_r, 6, 22))
                self.gui.circle(ee_r, radius=halo_r, color=0xFFFFFF)

            # 5) Arms (right then left, as original)
            self.roller_radius = float(massager.roller_radius)
            theta_r = self._scalar_from_1d(massager.thetaright)
            theta_l = self._scalar_from_1d(massager.thetaleft)
            self._draw_dual_arm(massager.base_x, massager.base_y, theta_r, massager.L1, ee_r)
            self._draw_dual_arm(massager.base_x, massager.base_y, theta_l, massager.L1, ee_l)

            # 6) HUD (kept)
            self.gui.text(f'Time: {massager.time_t:.3f} s', pos=self._hud_pos_time, color=0xFFFFFF)
            self.gui.text(f'Force(L): {cf_l:.2f} N', pos=self._hud_pos_fL,   color=0xFFFFFF)
            self.gui.text(f'Force(R): {cf_r:.2f} N', pos=self._hud_pos_fR,   color=0xFFFFFF)

        elif self.massager_tyepe == "straight":
            # contact halo
            ee   = massager.roller_center[0].to_numpy()
            cf   = RenderUtils.force_mag(massager.contact_force[0])
            halo_r = int(np.clip(6 + 0.004 * cf, 6, 22))
            self.gui.circle(ee, radius=halo_r, color=0xFFFFFF)

            # arms
            self.roller_radius = float(massager.roller_radius)
            box_size = massager.BOX_HX
            self.draw_straightarm(massager.base_x, massager.base_y, massager.theta0, massager.L_arm, massager.roller_center,box_size)

            # HUD
            self.gui.text(f'Time: {massager.time_t:.3f} s',
                    pos=(0.02, 0.95), color=0xFFFFFF)
            self.gui.text(f'Force(L): {float(np.linalg.norm(massager.contact_force[0])):.2f} N',
                    pos=(0.02, 0.92), color=0xFFFFFF)
            
        self.gui.show()

    # ──────────────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────────────
    def _end_eff_and_force(
        self,
        ee_center,  # (1,2) field/np
        f_vec       # (1,2) field/np
    ) -> Tuple[Optional[np.ndarray], float]:
        if ee_center is None or f_vec is None:
            return None, 0.0

        ee_np = RenderUtils.to_numpy_safe(ee_center).reshape(-1, 2)[0]
        mag   = RenderUtils.force_mag(RenderUtils.to_numpy_safe(f_vec).reshape(-1, 2)[0])
        return ee_np, mag

    @staticmethod
    def _scalar_from_1d(x: ArrayLike) -> float:
        if hasattr(x, "to_numpy"):
            a = x.to_numpy()
            return float(a.reshape(-1)[0])
        a = np.asarray(x)
        return float(a.reshape(-1)[0])

    def _draw_dual_arm(
        self,
        base_x: float,
        base_y: float,
        theta0: float,
        L: float,
        ee: Optional[np.ndarray],
    ) -> None:
        base_pt = np.array([base_x, base_y], dtype=np.float32)
        j2_dir  = np.array([np.sin(theta0), -np.cos(theta0)], dtype=np.float32)
        j2      = base_pt + j2_dir * L

        if ee is None:
            # If EE not available, skip draw (modality preserved: no arm end segment)
            return

        # Shadow
        off = self._arm_shadow
        self.gui.line(begin=base_pt + off, end=j2 + off, radius=3, color=0x151515)
        self.gui.line(begin=j2 + off,      end=ee + off, radius=3, color=0x151515)
        self.gui.circle(base_pt + off, radius=5, color=0x151515)
        self.gui.circle(j2 + off,      radius=5, color=0x151515)

        # Main body
        self.gui.line(begin=base_pt, end=j2, radius=3, color=ARM_BODY)
        self.gui.line(begin=j2, end=ee, radius=3, color=ARM_BODY)
        self.gui.circle(base_pt, radius=5, color=JOINT_COLOR)
        self.gui.circle(j2,      radius=5, color=JOINT_COLOR)

        # Edge highlight
        lit = self._edge_lit
        self.gui.line(begin=base_pt + lit, end=j2 + lit, radius=1, color=ARM_EDGE)
        self.gui.line(begin=j2 + lit, end=ee + lit, radius=1, color=ARM_EDGE)

        # Roller ring (kept)
        rr = int(self.roller_radius * self.resolution)
        self.gui.circle(ee, radius=rr + 2, color=ROLLER_EDGE)
        self.gui.circle(ee, radius=rr,     color=0xAAAAAA)


    def draw_straightarm(self,base_x, base_y, theta0, L, ee_center,box_size):
        base_pt = np.array([base_x, base_y], dtype=np.float32)
        tip     = ee_center[0].to_numpy()

        # single fixed link (base -> tip)
        j2 = base_pt + np.array([np.sin(theta0), -np.cos(theta0)], dtype=np.float32) * L

        # shadow
        off = np.array([0.003, -0.003], dtype=np.float32)
        self.gui.line(begin=base_pt + off, end=j2 + off, radius=3, color=0x151515)
        self.gui.circle(base_pt + off, radius=5, color=0x151515)

        # main body
        self.gui.line(begin=base_pt, end=j2, radius=3, color=ARM_BODY)
        self.gui.circle(base_pt, radius=5, color=JOINT_COLOR)

        # edge highlight
        lit = np.array([+0.0015, +0.0015], dtype=np.float32)
        self.gui.line(begin=base_pt + lit, end=j2 + lit, radius=1, color=ARM_EDGE)

        # roller ring at the tip
        rr = int(self.roller_radius * self.resolution)
        # self.draw_square(tip,(box_size,box_size),color=0xAAAAAA,radius=2)
        self.gui.circle(tip, radius=rr + 2, color=ROLLER_EDGE)
        self.gui.circle(tip, radius=rr,     color=0xAAAAAA)


    def draw_square(self, center, half, color=0xFFFFFF, radius=2):
        cx, cy = float(center[0]), float(center[1])
        hx, hy = float(half[0]), float(half[1])
        p1 = (cx - hx, cy - hy)
        p2 = (cx + hx, cy - hy)
        p3 = (cx + hx, cy + hy)
        p4 = (cx - hx, cy + hy)
        self.gui.line(begin=p1, end=p2, radius=radius, color=color)
        self.gui.line(begin=p2, end=p3, radius=radius, color=color)
        self.gui.line(begin=p3, end=p4, radius=radius, color=color)
        self.gui.line(begin=p4, end=p1, radius=radius, color=color)