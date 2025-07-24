import enum
from functools import lru_cache
from typing import NamedTuple

import torch


class VTBackend(enum.IntEnum):
    CPU    = 0
    GPU    = 1
    CUDA   = 2
    VULKAN = 3
    METAL  = 4
    OPENGL = 5

    def __str__(self):
        return f"gs.{self.name.lower()}"


class DeviceInfo(NamedTuple):
    torch_device: torch.device
    name: str
    total_mem_gb: float
    backend: VTBackend


class EnvConfig:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self._initialized = False
        self.theme = "dark"
        self.eps = 1e-15
        self.backend = VTBackend.CPU

    def init(self, *, theme: str = "dark", eps: float = 1e-15, backend: VTBackend = None):
        if self._initialized:
            raise RuntimeError("Environment already initialized.")
        self.theme = theme
        self.eps = eps
        if backend is not None:
            self.backend = backend

        device_info = get_device(self.backend)
        print(f"Running on ~~<[ {device_info.name} ]>~~ "
              f"with backend ~~<{device_info.backend}>~~. "
              f"Device memory: ~~<{device_info.total_mem_gb:.2f}>~~ GB.")

        self.torch_device = device_info.torch_device
        self._initialized = True
        return {
            "theme": self.theme,
            "eps": self.eps,
            "backend": self.backend,
        }


@lru_cache(maxsize=None)
def get_device(backend: VTBackend) -> DeviceInfo:
    # Helper for CPU fallback
    def cpu_info():
        cpu_dev = torch.device("cpu")
        # name = cpuinfo.get_cpu_info()["brand_raw"]  # optional detail
        total_mem = torch.cuda.mem_get_info()[1] / 1024**3 if torch.cuda.is_available() else 0.0
        return DeviceInfo(cpu_dev, "CPU", total_mem, VTBackend.CPU)

    if backend is VTBackend.CUDA:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        idx = torch.cuda.current_device()
        prop = torch.cuda.get_device_properties(idx)
        return DeviceInfo(torch.device("cuda", idx), prop.name, prop.total_memory / 1024**3, backend)

    if backend is VTBackend.METAL:
        if not torch.backends.mps.is_available():
            raise RuntimeError("Metal (MPS) not available")
        # on macOS MPS sits alongside CPU
        base = get_device(VTBackend.CPU)
        return DeviceInfo(torch.device("mps"), base.name, base.total_mem_gb, backend)

    if backend is VTBackend.VULKAN:
        if torch.cuda.is_available():
            return get_device(VTBackend.CUDA)
        # torch.xpu for Intel XPU support
        if hasattr(torch, "xpu") and torch.xpu.is_available():  
            idx = torch.xpu.current_device()
            prop = torch.xpu.get_device_properties(idx)
            return DeviceInfo(torch.device("xpu", idx), prop.name, prop.total_memory / 1024**3, backend)
        return cpu_info()

    if backend is VTBackend.GPU:
        # choose best available GPU‐style backend
        if torch.cuda.is_available():
            return get_device(VTBackend.CUDA)
        if torch.backends.mps.is_available():
            return get_device(VTBackend.METAL)
        return get_device(VTBackend.VULKAN)

    # fallback to CPU
    return cpu_info()
