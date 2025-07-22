# VETRIS: ViscoElastic Tissue-Robot Interaction Simulation with Material Point Method

**VETRIS** is a physics engine designed to simulate **viscoelastic soft tissue interactions with robotic systems** using the **Material Point Method (MPM)**. It focuses on capturing realistic tissue dynamics, viscoelastic deformation, and two-way coupling between soft and rigid bodies, enabling research in **soft robotics, biomechanics, and contact-aware manipulation**.

---

## **Abstract**

Robotic manipulation of soft biological tissues requires accurate modeling of complex viscoelastic behavior, non-linear deformations, and robust contact dynamics. **VETRIS** addresses these challenges by combining the Material Point Method (MPM) with advanced viscoelastic constitutive models to simulate tissue deformation under robotic interaction.

The engine includes:
- **Two-way coupling between rigid robots and viscoelastic tissues**.
- **Contact modeling with friction and rolling dynamics** for tool-tissue interaction.
- **Stable implicit time-stepping** to handle large deformations and high stiffness.
- **Modular design for constitutive models** (Neo-Hookean, Kelvin–Voigt, and extended viscoelastic laws).

---

## **Key Features**

- **Material Point Method Core**
  - 3D MPM solver with PIC/FLIP-style particle-grid transfer.
  - Support for both elastic and viscoelastic materials.

- **Robot-Tissue Interaction**
  - Two-way force exchange between a robotic manipulator (rigid) and deformable tissues.
  - Collision detection and frictional contact modeling.

- **Viscoelastic Tissue Models**
  - Neo-Hookean + damping-based viscoelasticity.
  - Extendable to elasto-plasticity and continuum damage models.

- **Rendering**
  - Real-time visualization using **PyRender** + **Trimesh**.
  - Particle-to-mesh reconstruction (convex hull or marching cubes).

- **Research-Focused**
  - Designed for robotic contact studies, surgical simulation, and bio-robotics research.

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

### 4. (Optional) GPU acceleration

If you have an NVIDIA GPU and CUDA installed, enable the Taichi CUDA backend in `vetris/config/config.yaml`:
```yaml
backend: cuda
```