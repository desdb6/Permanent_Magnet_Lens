# Permanent Magnet Electron Lens
**A compact and open-source permanent magnet electron lens: from simulation to realisation**

*Bachelor's thesis — Des De Borger, supervised by Prof. Dr. Johan Verbeeck*  
*EMAT, Department of Physics, University of Antwerp, 2025–2026*

---

## Overview

This repository contains the simulation toolbox and CAD files developed for the design, optimisation, and experimental validation of a miniature electron lens based on a permanent Nd₂Fe₁₄B ring magnet with two ferromagnetic AISI 1018 pole pieces. The lens was tested in a 30 kV Tescan MIRA scanning electron microscope, producing magnified images with a magnification of ×6.

The goal is to contribute to the development of compact, energy-efficient, and open-source electron microscopes with a significantly lower cost threshold than conventional solenoid-based systems.

---

## Repository structure

```
├── sim.py                   # Full simulation toolbox (Lens class + helper functions)
├── NdFeB_BH_curve.csv       # BH-curve data for grade N35 Nd₂Fe₁₄B magnet
├── cad/
│   ├── polepiece.FCStd      # FreeCAD parametric model of the pole pieces
│   ├── polepiece.step       # STEP file for CNC manufacturing
│   ├── lens_holder.blend    # Blender model of the 3D-printed lens holder
│   ├── grid_holder.blend    # Blender model of the 3D-printed grid holder
│   └── aperture.blend       # Blender model of the 3D-printed aperture
└── README.md
```

---

## Physics background

The lens consists of a homogeneously magnetised Nd₂Fe₁₄B ring magnet combined with two axially symmetric pole pieces. The pole pieces concentrate the magnetic field into the air gap at the centre of the lens, producing a strongly peaked, cylindrically symmetric field B(z) along the optical axis.

The field is modelled analytically by treating the pole-piece surfaces bordering the air gap as homogeneously charged magnetic surface charge distributions (annuli with charge density ±σ_M). The on-axis field is:

$$B(z) = \frac{B_m A_m}{A_g} \cdot \frac{1}{2}\left[\left(z+\frac{d}{2}\right)\left(\frac{1}{\sqrt{(z+d/2)^2+R_1^2}}-\frac{1}{\sqrt{(z+d/2)^2+R_2^2}}\right) - \left(z-\frac{d}{2}\right)\left(\frac{1}{\sqrt{(z-d/2)^2+R_1^2}}-\frac{1}{\sqrt{(z-d/2)^2+R_2^2}}\right)\right]$$

where R₁ and R₂ are the inner and outer radii of the pole-piece surfaces, d is the air gap width, and the prefactor accounts for demagnetisation via the permeance coefficient P_ci.

Electron trajectories are computed by numerically solving the relativistically corrected paraxial ray equation:

$$\frac{d^2r}{dz^2} + \frac{\eta^2 B(z)^2}{4V} r = 0, \quad V = \Phi\left(1 + \frac{e}{2m_e c^2}\Phi\right)$$

using an 8th-order Runge-Kutta method (`DOP853` via `scipy.integrate.solve_ivp`).

---

## Installation

```bash
git clone https://github.com/desdb6/Permanent_Magnet_Lens.git
cd Permanent_Magnet_Lens
pip install numpy scipy matplotlib
```

**Requirements:** Python 3.9+, NumPy, SciPy, Matplotlib

The BH-curve file `NdFeB_BH_curve.csv` must be present in the working directory for the `Lens` class to function.

---

## Quick start

```python
from sim import Lens, Mesh

# Define lens geometry (all lengths in mm, field in T, voltage in V)
lens = Lens(
    R_1=0.8,                    # Inner radius of pole-piece surface (mm)
    R_2=3.25,                   # Outer radius of pole-piece surface (mm)
    R_1_magnet=4.5,             # Inner radius of ring magnet (mm)
    R_2_magnet=6.0,             # Outer radius of ring magnet (mm)
    d=0.8,                      # Air gap width (mm)
    d_magnet=2.0,               # Magnet thickness (mm)
    B_r_magnet=1.17,            # Effective remanent field (T)
    B_r_magnet_theoretical=1.17,# Theoretical remanent field for BH-curve scaling (T)
    T=30e3,                     # Accelerating voltage (V)
    setup_length=228,           # Total column length (mm)
    lens_position=114,          # Lens position along column (mm)
    n=10000                     # Number of evaluation points
)

# Compute and display lens properties
lens.update_B_r_yoke()          # Compute effective field accounting for demagnetisation
lens.update_T()                 # Apply relativistic correction
lens.calculate_B_field()        # Compute B(z) along the optical axis
lens.calculate_lens_properties()# Compute f, Z_Fi, Z_Pi via ray tracing
lens.calculate_aberration_coeff()
lens.display_properties()       # Print results and plot ray diagram
```

**Output:**
```
Maximum flux density = 1.120 T
Pci = 2.575
Z_Fi = 11.377 mm
Z_Pi = -0.060 mm
f    = 11.437 mm
D    = 1.22 mm^-2
```

---

## Key features

### Interactive B-field explorer
Adjust all lens parameters with sliders and see B(z) update in real time. Press Enter to proceed to lens simulation with the selected parameters.

```python
from sim import make_lens_interactive
make_lens_interactive()
```

### Parameter variation plots
Plot how lens properties (f, Z_Fi, Z_Pi, P_ci, B_max) vary as a function of each geometry parameter individually.

```python
lens.variable_R_1(R_1_min=0.3, R_1_max=3.0, R_1_n=50)
lens.variable_R_2(R_2_min=1.0, R_2_max=6.0, R_2_n=50)
lens.variable_d(d_min=0.3, d_max=3.0, d_n=50)
lens.variable_B_r(B_r_min=0.8, B_r_max=1.4, B_r_n=50)
lens.variable_T(T_min=20, T_max=40, T_n=50)   # voltage in kV
```

Distortion coefficient variants are also available:
```python
lens.variable_R_1_ab(0.3, 3.0, 50)
lens.variable_R_2_ab(1.0, 6.0, 50)
lens.variable_d_ab(0.3, 3.0, 50)
lens.variable_B_r_ab(0.8, 1.4, 50)
```

### Ray tracing and image formation
```python
# Set up object position and height
lens.setup_parameters(object_pos=94, object_height=1.5, lens_pos=101.08)

# Plot ray diagram with custom initial conditions
import numpy as np
initial_values = np.array([[1, 0], [0, 0.01], [-1, 0]])
lens.plot_setup(initial_values=initial_values)
```

### Monte Carlo shadowgraph simulation
Simulate the rear mesh shadowgraph experiment used to measure the focal length experimentally. Runs continuously until interrupted with Ctrl+C.

```python
mesh = Mesh(pos=144.22, line_dist=254e-3, line_thickness=50e-3)
lens.add_mesh(mesh)

lens.monte_carlo(
    object_height=0,
    opening_angle=100e-3,
    pixel_size=55e-3,       # AdvaPIX/MiniPIX pixel size (mm)
    camera_pos=228,
    pixel_count=256,
    voxel_length=0.05
)
```

### BH-curve and operating point
```python
from sim import BH_curve_magnet, plot_operating_point

spline = BH_curve_magnet(B_max=1.17)  # Scale BH-curve to target B_r
plot_operating_point(spline, reluctance_correction=0.39)
```

---

## Optimal lens parameters and results

| Parameter | Symbol | Value |
|---|---|---|
| Inner pole-piece radius | R₁ | 0.8 mm |
| Outer pole-piece radius | R₂ | 3.25 mm |
| Air gap width | d_g | 0.8 mm |
| Magnet inner radius | R₁_m | 4.5 mm |
| Magnet outer radius | R₂_m | 6.0 mm |
| Magnet thickness | d_m | 2.0 mm |
| Remanent field (N35 grade) | B_r | 1.17 T |

| Simulated property | Value |
|---|---|
| Focal length f_sim | 11.437 mm |
| Image principal plane z_Pi | −0.060 mm |
| Permeance coefficient P_ci | 2.575 |
| Maximum flux density B_max | 1.120 T |
| Distortion coefficient D | 1.22 mm⁻² |

**Experimentally measured focal length (30 kV, Tescan MIRA SEM):**  
f_exp = 25 ± 2 mm

The discrepancy is attributed to magnetic flux leakage. A numerical validation shows that scaling the field by a factor of 0.67 reproduces the experimental focal length, giving a corrected simulation of f = 24.6 mm.

---

## CAD files and manufacturing

The pole pieces were manufactured from AISI 1018 steel (surface passivated) by [Protolabs Network](https://www.hubs.com) using CNC machining. The STEP file can be uploaded directly to any CNC on-demand service.

The lens holder, grid holder, and aperture were 3D-printed in conductive PLA (resistivity ~450–6000 kΩ measured) to prevent charge buildup inside the electron microscope column. Blender source files are included.

---

## Notes

- All lengths are in **millimetres** throughout the codebase
- `B_r_magnet` scales the BH-curve so that B(H=0) equals the supplied value; it does not assume a fixed linear permeability
- The `leak_factor` parameter in the `__main__` block can be used to model flux leakage (e.g. `leak_factor=0.67` to match the experimental result)
- `NdFeB_BH_curve.csv` must use semicolons as delimiters and commas as decimal separators (European format), as exported from the original data source

---

## Citing this work

If you use this toolbox or CAD files in your research, please cite:

```
De Borger, D. (2026). A compact and open-source permanent magnet electron lens:
from simulation to realisation. Bachelor's thesis, University of Antwerp.
Supervisor: Prof. Dr. J. Verbeeck. EMAT, Department of Physics.
```

---

## License

MIT License — free to use, modify, and distribute with attribution.

---

## Contact

Des De Borger — [Des.deborger@student.uantwerpen.be](mailto:Des.deborger@student.uantwerpen.be)  
EMAT | Electron Microscopy for Materials Science, University of Antwerp
