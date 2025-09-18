import imageio

class Recorder:
    def __init__(self, cfg):
        self.cfg = cfg.flags.record
        self.output_file = getattr(self.cfg, "output_file", "output.gif")
        self.fps = getattr(self.cfg, "fps", 30)
        self.frames = []

    def add_frame(self, frame):
        """
        Add a frame to the recorder.
        frame should be a NumPy array (e.g., from GUI.get_image() or a screenshot).
        """
        self.frames.append(frame)

    def save(self):
        """Write the accumulated frames to the video file."""
        imageio.mimsave(self.output_file, self.frames, fps=self.fps)