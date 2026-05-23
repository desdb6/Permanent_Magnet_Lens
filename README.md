# Permanent Magnet Electron Lens
**A compact and open-source permanent magnet electron lens: from simulation to realisation**

*Bachelor's thesis — Des De Borger, supervised by Prof. Dr. Johan Verbeeck*
*EMAT, Department of Physics, University of Antwerp, 2025–2026*

---

## Overview

This repository contains the full simulation toolbox and CAD files developed for the design, optimisation, and experimental validation of a miniature electron lens based on a permanent Nd₂Fe₁₄B ring magnet and two ferromagnetic AISI 1018 pole pieces. The lens was tested in a 30 kV Tescan MIRA scanning electron microscope.

The goal is to contribute to the development of compact, energy-efficient, and open-source electron microscopes with a significantly lower cost threshold than conventional solenoid-based systems.

---

## Repository structure

```
├── simulation/
│   ├── toolbox.py               # Main simulation toolbox
│   ├── magnetic_field.py        # Analytical B(z) model and derivatives
│   ├── paraxial_solver.py       # Numerical paraxial ray equation solver (RK8)
│   ├── distortion.py            # Distortion coefficient calculation
│   ├── monte_carlo.py           # Monte Carlo shadowgraph experiment simulator
│   └── variation_plots.py       # Parameter variation and optimisation plots
├── cad/
│   ├── polepiece.FCStd          # FreeCAD parametric model of the pole pieces
│   ├── polepiece.step           # STEP file for CNC manufacturing
│   ├── lens_holder.blend        # Blender model of the 3D-printed lens holder
│   ├── grid_holder.blend        # Blender model of the 3D-printed grid holder
│   └── aperture.blend           # Blender model of the 3D-printed aperture
├── examples/
│   ├── basic_simulation.py      # Minimal working example
│   ├── optimisation_example.py  # Full optimisation workflow
│   └── shadowgraph_example.py   # Focal length measurement simulation
└── README.md
```

---

## Physics background

The lens consists of a homogeneously magnetised Nd₂Fe₁₄B ring magnet combined with two axially symmetric pole pieces. The pole pieces concentrate the magnetic field into the air gap at the centre of the lens, producing a strongly peaked, cylindrically symmetric field B(z) along the optical axis.

The field is modelled analytically by treating the pole-piece surfaces bordering the air gap as homogeneously charged magnetic surface charge distributions (annuli with surface charge density ±σ_M). The on-axis field is then:

$$B(z) = \frac{B_m A_m}{A_g} \cdot \frac{1}{2}\left[\left(z+\frac{d}{2}\right)\left(\frac{1}{\sqrt{(z+d/2)^2+R_1^2}}-\frac{1}{\sqrt{(z+d/2)^2+R_2^2}}\right) - \left(z-\frac{d}{2}\right)\left(\frac{1}{\sqrt{(z-d/2)^2+R_1^2}}-\frac{1}{\sqrt{(z-d/2)^2+R_2^2}}\right)\right]$$

where R₁ and R₂ are the inner and outer radii of the pole-piece surfaces, d is the air gap width, and B_m A_m / A_g accounts for demagnetisation via the permeance coefficient.

Electron trajectories are computed by numerically solving the paraxial ray equation:

$$\frac{d^2r}{dz^2} + \frac{\eta^2 B(z)^2}{4V} r = 0$$

using an 8th-order Runge-Kutta method.

---

## Installation

```bash
git clone https://github.com/desdb6/Permanent_Magnet_Lens.git
cd Permanent_Magnet_Lens
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, NumPy, SciPy, Matplotlib

---

## Quick start

```python
from simulation.toolbox import PermanentMagnetLens

# Define lens geometry
lens = PermanentMagnetLens(
    R1=0.8e-3,      # Inner radius of pole-piece surface (m)
    R2=3.25e-3,     # Outer radius of pole-piece surface (m)
    d_g=0.8e-3,     # Air gap width (m)
    R1_m=4.5e-3,    # Inner radius of magnet (m)
    R2_m=6.0e-3,    # Outer radius of magnet (m)
    d_m=2.0e-3,     # Magnet thickness (m)
    B_r=1.17,       # Remanent field of magnet (T)
    voltage=30e3    # Accelerating voltage (V)
)

# Compute lens properties
print(lens.focal_length())        # Asymptotic focal length (m)
print(lens.principal_plane())     # Image principal plane position (m)
print(lens.distortion_coeff())    # Distortion coefficient (m⁻²)

# Plot the magnetic field
lens.plot_field()

# Interactive parameter exploration
lens.interactive_plot()
```

---

## Lens parameters and simulation results

The following optimal parameter set was determined using the simulation toolbox:

| Parameter | Value |
|---|---|
| Inner pole-piece radius R₁ | 0.8 mm |
| Outer pole-piece radius R₂ | 3.25 mm |
| Air gap width d_g | 0.8 mm |
| Magnet inner radius R₁_m | 4.5 mm |
| Magnet outer radius R₂_m | 6.0 mm |
| Magnet thickness d_m | 2.0 mm |
| Remanent field B_r (N35 grade) | 1.17 T |

| Simulated property | Value |
|---|---|
| Focal length f_sim | 11.437 mm |
| Image principal plane z_Pi | −0.060 mm |
| Permeance coefficient P_ci | 2.575 |
| Maximum flux density B_max | 1.120 T |
| Distortion coefficient D | 1.22 mm⁻² |

**Experimentally measured focal length:** f_exp = 25 ± 2 mm (30 kV, Tescan MIRA SEM)

The discrepancy is attributed to flux leakage, which reduces the effective field in the air gap to approximately 67% of the analytically predicted value.

---

## CAD files and manufacturing

The pole pieces were manufactured from AISI 1018 steel by [Protolabs Network](https://www.hubs.com) using CNC machining, with a passivated surface finish to prevent corrosion. The STEP file is provided for direct upload to any CNC on-demand service.

The lens holder, grid holder, and aperture were 3D-printed in conductive PLA to prevent charge buildup inside the electron microscope. The Blender source files are included.

---

## Citing this work

If you use this toolbox or CAD files in your research, please cite:

```
De Borger, D. (2026). A compact and open-source permanent magnet electron lens:
from simulation to realisation. Bachelor's thesis, University of Antwerp.
Supervisor: Prof. Dr. J. Verbeeck.
```

---

## License

MIT License — free to use, modify, and distribute with attribution.

---

## Contact

Des De Borger — [Des.deborger@student.uantwerpen.be](mailto:Des.deborger@student.uantwerpen.be)
EMAT | Electron Microscopy for Materials Science, University of Antwerp
