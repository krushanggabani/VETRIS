import numpy as np


def hexrgb(h: int):
    return ((h >> 16) & 255, (h >> 8) & 255, h & 255)


tissue_palettes = {
    "muscle":   {"low": np.array([139,   0,   0], dtype=np.float32),
                 "mid": np.array([205,  92,  92], dtype=np.float32),
                 "high":np.array([255, 182, 193], dtype=np.float32)},
    "organic":  {"low": np.array([102,  51,   0], dtype=np.float32),
                 "mid": np.array([153, 102,  51], dtype=np.float32),
                 "high":np.array([255, 224, 189], dtype=np.float32)},
    "vascular": {"low": np.array([120,   0,   0], dtype=np.float32),
                 "mid": np.array([200,   0,   0], dtype=np.float32),
                 "high":np.array([255,  99,  71], dtype=np.float32)},
    "surgical": {"low": np.array([178, 102, 102], dtype=np.float32),
                 "mid": np.array([255, 160, 160], dtype=np.float32),
                 "high":np.array([255, 230, 230], dtype=np.float32)},
    "classic":  {"low": np.array(hexrgb(0xC74A4A), dtype=np.float32),
                 "mid": np.array(hexrgb(0xF59A9A), dtype=np.float32),
                 "high":np.array(hexrgb(0xF7C6C6), dtype=np.float32)},
}