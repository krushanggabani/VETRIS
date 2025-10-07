import time
import numpy as np
import taichi as ti

from vetris.engine.mpm import mpmengine
from vetris.io.recorder import Recorder
from vetris.io.logger import Logger

try:
    # Only import renderer if we actually need it (avoids GUI deps in headless)
    from vetris.vis.render import Renderer
except Exception:
    Renderer = None


class Simulation:
    """
    Minimal driver with proper headless modality and safe loop/stop conditions.
    """
    def __init__(
        self,
        cfg,
        *,
        headless: bool = True,
        use_gpu: bool = True,
        device_memory_GB: float | None = None,
        fast_math: bool = True,
        debug_ti: bool = False,
        taichi_reinit: bool = True,
        seed: int | None = None,
    ):
        # ------------------- config & mode -------------------
        self.cfg = cfg
        self.headless = bool(headless)
        self.use_gpu = bool(use_gpu)
        self.fast_math = bool(fast_math)
        self.debug_ti = bool(debug_ti)
        self.device_memory_GB = device_memory_GB

        # Optional deterministic seed
        if seed is not None:
            np.random.seed(int(seed))

        # ------------------- taichi init ---------------------
        # Safe re-init (useful if you create multiple Simulation() in one process)
        if taichi_reinit:
            ti.reset()

        init_kwargs = dict(
            arch=ti.gpu if self.use_gpu else ti.cpu,
            debug=self.debug_ti,
            fast_math=self.fast_math,
        )
        if self.device_memory_GB is not None:
            # Not all Taichi builds accept device_memory_GB; guard it
            init_kwargs["device_memory_GB"] = float(self.device_memory_GB)

        ti.init(**init_kwargs)

        # ------------------- subsystems ----------------------
        self.recorder = Recorder(cfg)
        self.logger = Logger(cfg)

        # Only create the renderer if NOT headless and if available
        self.renderer = None
        if not self.headless:
            if Renderer is None:
                print("[Simulation] Renderer unavailable; forcing headless=True.")
                self.headless = True
            else:
                self.renderer = Renderer(cfg)

        # Physics engine (MPM)
        self.engine = mpmengine(cfg)

        # Derived stop time from the driver (massager)
        # If you use an override elsewhere, you can plumb it in and override here.
        try:
            self.stop_time = float(self.engine.massager.massager.Time_period)
        except Exception:
            # Fallback: a reasonable default if driver not initialized yet
            self.stop_time = 1.0

        # Debug
        # print("Simulation initialized; headless =", self.headless)

    # --------------- single frame (N substeps) ---------------
    def _substep_block(self, n_substeps: int) -> None:
        for _ in range(int(n_substeps)):
            # engine.run() returns False if unstable / early exit requested
            ok = self.engine.run()
            if not ok:
                break

    # ---------------------- main loop ------------------------
    def run(
        self,
        *,
        substeps_per_frame: int = 15,
        max_frames: int = 10_000,
        max_sim_time: float | None = None,
        render_every: int = 1,
        print_every: int = 0,  # set >0 to print debug line every N frames
    ):
        """
        Run the simulation loop.

        Args:
            substeps_per_frame: how many engine substeps per "frame"
            max_frames: hard cap on number of frames to avoid infinite loops
            max_sim_time: optional safety cap on simulated time (seconds)
            render_every: render frequency for non-headless runs
            print_every: if >0, print a debug line every N frames
        """
        # Respect explicit time cap if provided, else use massager period
        time_cap = float(max_sim_time) if max_sim_time is not None else float(self.stop_time)

        frame = 0
        start_wall = time.time()

        # Loop until (unstable) or (sim time >= time_cap) or (frame cap)
        while True:
            # Stop conditions
            t_now = float(self.engine.massager.massager.time_t)
            if t_now >= time_cap:
                break
            if frame >= int(max_frames):
                break
            if int(self.engine.unstable[None]) != 0:
                break

            # Do a block of substeps
            self._substep_block(substeps_per_frame)

            # Log state
            try:
                self.logger.log(self.engine.get_state())
            except Exception as e:
                print("[Simulation] Logger error:", e)

            # Render if GUI mode
            if (not self.headless) and (self.renderer is not None) and (frame % max(1, render_every) == 0):
                try:
                    self.renderer.render(self.engine)
                except Exception as e:
                    # Keep simulation going even if rendering fails
                    print("[Simulation] Render error:", e)

            # Optional debug line
            if print_every > 0 and (frame % print_every == 0):
                cf = self.engine.massager.massager.contact_force[0].to_numpy()
                cf_norm = float(np.linalg.norm(cf.reshape(-1))) if cf.size else 0.0
                print(f"[frame {frame:05d}] t={t_now:.5f}  |F|={cf_norm:.4e}  unstable={int(self.engine.unstable[None])}")

            frame += 1

        # You can save logs here or outside after run() returns
        self.logger.save_logs_to_csv("data/logs/simulation_logs.csv")

        wall_sec = time.time() - start_wall
        print(f"[Simulation] Finished. frames={frame}, sim_t={float(self.engine.massager.massager.time_t):.6f}, wall={wall_sec:.2f}s")

    # -------------------------- reset ------------------------
    def reset(self):
        """
        Reset or rebuild subsystems if you need to restart a simulation cleanly.
        Note: mpmengine.reset() was a no-op in your code; you can extend it.
        """
        # print("Resetting simulation...")
        try:
            self.engine.reset()
        except Exception:
            pass

    # -------------------------- close ------------------------
    def close(self):
        """
        Close I/O resources. (Renderer/Recorder/Logger cleanups.)
        """
        # print("Closing simulation...")
        try:
            self.recorder.close()
        except Exception:
            pass

        # If your Renderer holds a GUI/window, expose a close API and call it here:
        if self.renderer is not None and hasattr(self.renderer, "close"):
            try:
                self.renderer.close()
            except Exception:
                pass
