# VETRIS : ViscoElastic Tissue–Robot Interaction Simulation

VETRIS is a modular 2D physics engine in **Python + Taichi** for simulating viscoelastic soft-tissue interaction with robotic systems using **MLS-MPM**. It targets realistic tissue dynamics, viscoelastic deformation, and two-way soft–rigid coupling, with first-class support for **Real2Sim calibration**, **environment validation**, and **control/design optimization**.


![base_model](data/media/base_model.gif)


## 🌟 Features
- **MPM core** with clean APIs (P2G/Grid/G2P), stable time-stepping, and contact/friction models.
- **Viscoelastic materials**: Neo-Hookean, Kelvin–Voigt, Standard Linear Solid (plug-in ready).
- **Two-way coupling** between deformable tissue and rigid proxies (SDF-based contact).
- **Virtual sensors** (force, indentation, pose) aligned to real data streams.
- **Real2Sim calibration**: signal alignment, multi-metric losses, and optimizer backends.
- **Validation & optimization**: reusable metrics for scenario validation, controller tuning, and design studies.
- **Reproducible I/O**: structured logging, renderer overlays, and frame/video recording.

---

## 📦 Installation

### ❗ Prerequisites
- **Python 3.10+**
- (Optional) NVIDIA GPU + CUDA for accelerated Taichi backends


### 1. Clone the repository

```bash
git clone https://github.com/krushanggabani/VETRIS.git
cd VETRIS
```

### 2. Create and activate a virtual environment

#### Windows (PowerShell)
```powershell
python -m venv VRenv    
.\VRenv\Scripts\Activate.ps1 
```

#### Linux / macOS (bash)
```bash
python3 -m venv VRenv
source VRenv/bin/activate
```


### 3. Install package and dependencies

```bash
pip install -e .            # Installs in editable mode
```

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

Questions or feedback? Open an issue or email me at <krushgabani95@gmail.com>.

Happy simulating! 🚀