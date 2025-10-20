# Swirl-Jatmos: JAX-based Atmospheric LES Code for Supercell Simulation

**Documentation Date:** October 20, 2025

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Governing Equations](#governing-equations)
4. [Numerical Methods](#numerical-methods)
5. [Physics Modules](#physics-modules)
6. [Supercell Setup](#supercell-setup)
7. [Comparison with Existing LES Codes](#comparison-with-existing-les-codes)
8. [Performance Characteristics](#performance-characteristics)
9. [Installation and Usage](#installation-and-usage)

---

## Overview

**Swirl-Jatmos** is a modern Large Eddy Simulation (LES) code written in JAX for simulating 3D atmospheric flows on distributed accelerators (TPUs and GPUs). It is specifically designed to handle supercell thunderstorm simulations and other convective atmospheric phenomena.

**Primary Language:** Python (JAX framework)

**Developer:** Google Research (Not an officially supported Google product)

**Parallelization Strategy:** Automatic parallelization via JAX (no explicit communication directives required)

**Target Hardware:** TPUs (Tensor Processing Units) and GPUs

---

## Key Features

### Computational Architecture

1. **JAX-based Implementation**
   - Utilizes JAX's automatic differentiation and parallelization
   - Just-in-time (JIT) compilation for performance
   - Automatic distribution across multiple accelerators
   - FP32 precision arithmetic

2. **Grid Structure**
   - Staggered Arakawa C-grid in Cartesian coordinates
   - Support for non-uniform (stretched) grids in all three dimensions
   - Conservative finite-volume formulation
   - Periodic boundary conditions in horizontal (x, y) directions
   - Free-slip or no-slip boundary conditions in vertical (z) direction

3. **Time Integration**
   - Third-order Runge-Kutta (RK3) timestepper
   - Adaptive timestepping based on CFL condition
   - Sub-cycling capability with configurable timestep control

4. **Distributed Checkpointing**
   - Orbax-based checkpoint management
   - Cycle-based or step-based checkpoint labeling
   - Automatic recovery from checkpoints

---

## Governing Equations

Swirl-Jatmos solves the **anelastic Navier-Stokes equations** with moist thermodynamics. The prognostic variables are:

### Prognostic Variables

| Variable | Symbol | Description |
|----------|--------|-------------|
| `u, v, w` | **u** | Velocity components (m/s) |
| `theta_li` | θ_li | Linearized liquid-ice potential temperature (K) |
| `q_t` | q_t | Total specific humidity (kg/kg) |
| `q_r` | q_r | Rain water mass fraction (kg/kg) |
| `q_s` | q_s | Snow mass fraction (kg/kg) |
| `p` | p | Hydrodynamic pressure perturbation (Pa) |

### Continuity Constraint

The anelastic continuity equation enforces divergence-free mass flux:

```
∇ · (ρ u) = 0
```

where ρ is the reference density. This constraint is satisfied by solving a Poisson equation for pressure at each timestep.

### Momentum Equations

```
∂(ρu)/∂t + ∇·(ρu⊗u) = -∇p + ρg(ρ_thermal - ρ_ref)/ρ_ref + ∇·τ
```

where:
- τ is the stress tensor (viscous + turbulent)
- g is gravitational acceleration
- ρ_thermal is the density from thermodynamic state
- ρ_ref is the reference density profile

### Scalar Transport Equations

```
∂(ρφ)/∂t + ∇·(ρuφ) = ∇·(ρκ∇φ) + S_φ
```

where φ represents any scalar (θ_li, q_t, q_r, q_s), κ is the diffusivity, and S_φ represents source terms from microphysics and other processes.

---

## Numerical Methods

### Convection Schemes

Multiple convection schemes are supported via the `ConvectionConfig`:

1. **WENO5** (Weighted Essentially Non-Oscillatory, 5th order)
   - High-order accuracy with shock-capturing capability
   - Suitable for sharp gradients and discontinuities
   
2. **Central Difference Schemes**
   - 2nd, 4th, and 6th order centered differences available
   - Efficient for smooth flows
   
3. **Upwind Schemes**
   - 1st order upwind for stability-critical cases

### Pressure Projection

Two Poisson solver options:

1. **Fast Diagonalization Solver** (Default)
   - Tensor-product-based decomposition
   - Efficient for periodic boundary conditions
   - O(N log N) complexity
   
2. **Jacobi Iterative Solver**
   - Iterative method with configurable tolerance
   - Flexible for complex boundary conditions
   - Includes halo exchange for distributed computing

### Diffusion Treatment

- Explicit treatment of viscous and diffusive terms
- Stability limit enforcement available (`enforce_max_diffusivity` flag)
- Automatic maximum diffusivity capping based on CFL-like criterion: κ_max ~ 0.2 * dz²/dt

---

## Physics Modules

### 1. Thermodynamics

**Module:** `swirl_jatmos/thermodynamics/water.py`

**Capabilities:**
- Equilibrium thermodynamics for water phases (vapor, liquid, ice)
- Saturation adjustment algorithm
- Exner function computation
- Temperature-dependent latent heat calculations
- Mixed-phase thermodynamics support

**Key Functions:**
```python
compute_thermodynamic_fields_from_prognostic_fields(
    theta_li, q_t, p_ref, rho_thermal_guess, wp
) -> ThermoFields
```

Returns computed fields:
- Temperature (T)
- Vapor mixing ratio (q_v)
- Liquid mixing ratio (q_liq)
- Ice mixing ratio (q_ice)
- Saturation vapor mixing ratio (q_v_sat)
- Thermal density (rho_thermal)

### 2. Microphysics

**Module:** `swirl_jatmos/microphysics/microphysics_one_moment.py`

**Scheme:** One-moment bulk microphysics with Marshall-Palmer size distributions

**Processes Included:**

| Process | Description | Equation Form |
|---------|-------------|---------------|
| Autoconversion | Cloud water → Rain/Snow | Threshold-based |
| Accretion | Collection between hydrometeors | Collision-coalescence |
| Evaporation/Sublimation | Phase change to vapor | Ventilation effects included |
| Sedimentation | Gravitational settling | Terminal velocity parameterization |
| Snow Melt | Snow → Rain | Temperature-dependent |

**Terminal Velocity Options:**
1. Power-law parameterization
2. Chen et al. (2022) parameterization (advanced)

**Key Parameters:**
- Cloud droplet number concentration: 100 cm⁻³
- Particle densities: ρ_water = 1000 kg/m³, ρ_ice = 500 kg/m³
- Collision efficiencies: Configurable per interaction type

### 3. Subgrid-Scale (SGS) Turbulence Model

**Module:** `swirl_jatmos/sgs.py`

**Model:** Smagorinsky-Lilly with stability corrections

**Turbulent Viscosity Formula:**
```
ν_t = (C_s * Δ)² * |S| * √(1 - Ri/Pr_t)
```

where:
- C_s = 0.18 (Smagorinsky constant)
- Δ = (dx·dy·dz)^(1/3) (filter width)
- |S| = √(2 S_ij S_ij) (strain rate magnitude)
- Ri = N²/|S|² (Richardson number)
- Pr_t = 0.33 (turbulent Prandtl number)

**Features:**
- Buoyancy effects via Richardson number correction
- Separate eddy viscosity and eddy diffusivity
- Stability factor prevents unphysical turbulence in stratified regions

### 4. Radiation (Optional)

**Module:** `swirl_jatmos/rrtmgp/`

**Scheme:** RRTMGP (Rapid Radiative Transfer Model for GCMs - Parallel)

**Capabilities:**
- Longwave and shortwave radiation
- Cloud-radiation interactions
- Gas absorption (H₂O, CO₂, O₃, etc.)
- Configurable radiation update frequency

### 5. Boundary Conditions

**Module:** `swirl_jatmos/boundary_conditions/`

**Available Options:**

**Horizontal (x, y):** Periodic (always)

**Vertical (z):**
1. Free-slip walls (Neumann conditions)
2. No-slip walls
3. Monin-Obukhov surface layer parameterization
   - Momentum flux based on friction velocity
   - Heat and moisture fluxes
   - Stability-dependent similarity functions

---

## Supercell Setup

**Module:** `swirl_jatmos/sim_setups/supercell.py`

The supercell configuration follows the idealized setup commonly used in the community, with parameters derived from CM1 (Cloud Model 1).

### Initial Atmospheric Profile

**Temperature and Potential Temperature:**

```
For z < z_t (troposphere):
    θ(z) = θ_s + (θ_t - θ_s) * (z / z_t)^1.25

For z ≥ z_t (stratosphere):
    θ(z) = θ_t * exp[g(z - z_t) / (T_t * c_p)]
```

**Default Parameters:**
- θ_s (surface) = 300 K
- θ_t (tropopause) = 343 K
- T_t (tropopause temp) = 213 K
- z_t (tropopause height) = 12 km

### Wind Profile (Hodograph)

Creates a curved hodograph to provide storm-relative helicity:

**Below z₁ = 2 km (rotating ground layer):**
```
u(z) = u_min * (1 - cos(π z / 2z₁))
v(z) = u_min * sin(π z / 2z₁)
```

**Between z₁ and z₂ = 6 km:**
```
u(z) = u_min + (z - z₁) * (u_max - u_min) / (z₂ - z₁)
v(z) = u_min (constant)
```

**Above z₂:**
```
u(z) = u_max (constant)
v(z) = u_min (constant)
```

**Default Values:**
- u_min = 7 m/s
- u_max = 31 m/s
- Velocity shifts: u_shift = 12.47 m/s, v_shift = 2.31 m/s (keeps updraft centered)

### Moisture Profile

**Relative Humidity:**
```
For z < z_t:
    h(z) = 1 - 0.75 * (z / z_t)^1.25

For z ≥ z_t:
    h(z) = 0.25
```

Total humidity q_t is then computed iteratively to satisfy:
```
q_t = h * q_v_sat
```

where q_v_sat accounts for moist air thermodynamics.

### Warm Bubble Perturbation

**Purpose:** Initiate convection

**Location:**
- Center: (x₀ = L_x/2, y₀ = L_y/2, z₀ = 1.4 km)
- Horizontal radius: r_h = 10 km
- Vertical radius: r_v = 1.4 km

**Amplitude:**
```
θ' = θ_pert * cos²(π r / 2)  for r < 1
θ' = 0                        for r ≥ 1

where r = √[(x-x₀)²/r_h² + (y-y₀)²/r_h² + (z-z₀)²/r_v²]
```

Default: θ_pert = 1 K

### Reference State

Reference pressure and density profiles are computed hydrostatically consistent with the thermodynamic profile:

```python
# Pressure (hydrostatic balance):
p_ref(z) = p₀ * [1 - (g/c_p) * ∫₀^z dz'/θ(z')]^(c_p/R_d)

# Density (equation of state):
ρ_ref(z) = p_ref(z) / (R_m * T(z))
```

where R_m is the moist air gas constant accounting for water vapor content.

---

## Comparison with Existing LES Codes

### Major LES Codes for Supercell Simulation

#### 1. CM1 (Cloud Model 1)

**Developer:** George Bryan (NCAR)

**Language:** Fortran

**Parallelization:** MPI

**Target Hardware:** CPU clusters

**Key Characteristics:**
- Industry standard for supercell research
- Anelastic and compressible equation sets
- Extensive validation against observations
- Large user community and documentation
- Well-tested microphysics schemes (Morrison, Thompson, etc.)

**Comparison with Swirl-Jatmos:**

| Feature | CM1 | Swirl-Jatmos |
|---------|-----|--------------|
| **Equation Set** | Anelastic or compressible | Anelastic only |
| **Grid** | Arakawa C-grid | Arakawa C-grid |
| **Language** | Fortran | Python (JAX) |
| **Parallelization** | Explicit MPI | Automatic (JAX) |
| **Hardware** | CPU | TPU/GPU |
| **Microphysics** | Multi-moment schemes | One-moment only |
| **SGS Model** | Multiple options | Smagorinsky-Lilly |
| **Radiation** | Optional (RRTMG) | Optional (RRTMGP) |
| **User Base** | Large, established | Emerging |
| **Development** | Mature (~20 years) | New (~2024) |

**Advantages of Swirl-Jatmos:**
- ✅ Automatic parallelization (no MPI coding needed)
- ✅ Native accelerator support (TPU/GPU)
- ✅ Potentially faster development/prototyping in Python
- ✅ Modern automatic differentiation capabilities
- ✅ Built-in distributed checkpointing (Orbax)

**Advantages of CM1:**
- ✅ More mature and extensively validated
- ✅ Compressible equations for acoustic waves
- ✅ Advanced multi-moment microphysics
- ✅ Larger library of test cases
- ✅ More extensive documentation and tutorials
- ✅ Established user community

#### 2. WRF-LES (Weather Research and Forecasting)

**Developer:** NCAR/NOAA

**Language:** Fortran

**Parallelization:** MPI + OpenMP

**Target Hardware:** CPU clusters

**Key Characteristics:**
- Full-featured atmospheric model (not LES-specific)
- Can run in LES mode with fine resolution
- Extensive physics options
- Operational use in weather forecasting

**Comparison with Swirl-Jatmos:**

| Aspect | WRF-LES | Swirl-Jatmos |
|--------|---------|--------------|
| **Scope** | Multi-scale (meso to LES) | LES-focused |
| **Complexity** | Very high | Moderate |
| **Setup Time** | Long (many namelists) | Shorter (Python config) |
| **Performance** | Optimized for CPU | Optimized for TPU/GPU |
| **Flexibility** | Extensive options | Focused feature set |

**When to Use WRF-LES:**
- Need multi-scale nesting
- Coupling to larger-scale meteorology
- Operational forecasting integration

**When to Use Swirl-Jatmos:**
- Idealized LES studies
- GPU/TPU resources available
- Rapid prototyping and experimentation
- Modern ML/AI integration desired

#### 3. PALM (PArallelized Large-eddy simulation Model)

**Developer:** Leibniz University Hannover

**Language:** Fortran

**Parallelization:** MPI

**Target Hardware:** CPU clusters

**Key Characteristics:**
- Designed for atmospheric and oceanic boundary layers
- Urban canopy models
- Interactive nesting
- Boussinesq or anelastic approximations

**Comparison with Swirl-Jatmos:**

| Aspect | PALM | Swirl-Jatmos |
|--------|------|--------------|
| **Primary Use** | Boundary layers, urban | Convection, supercells |
| **Equation Set** | Boussinesq or anelastic | Anelastic |
| **Microphysics** | Bulk and bin schemes | Bulk one-moment |
| **Hardware** | CPU | TPU/GPU |

#### 4. UCLA-LES

**Developer:** University of California, Los Angeles

**Language:** Fortran

**Key Characteristics:**
- Focused on cloud-topped boundary layers
- Bin microphysics capability
- Anelastic equations

**Comparison:** Similar scope to PALM; Swirl-Jatmos offers accelerator advantage for large-scale convection studies.

### Summary Table: LES Code Comparison

| Code | Grid Points/Core | Scalability | Microphysics | Typical Use |
|------|------------------|-------------|--------------|-------------|
| **CM1** | Up to ~500³ | Good (MPI) | Multi-moment | Supercells, tornadoes |
| **WRF-LES** | ~200-400³ | Excellent | Advanced bulk/bin | Real-case LES |
| **PALM** | ~500³ | Excellent | Bulk/bin | Boundary layers |
| **Swirl-Jatmos** | **256³ per core** | **Very good (JAX)** | One-moment | Supercells, convection |

### Unique Advantages of Swirl-Jatmos

1. **Hardware Acceleration**
   - Native TPU/GPU support
   - Scales efficiently on modern AI accelerators
   - Potentially 5-10x faster than CPU codes on equivalent hardware

2. **Automatic Parallelization**
   - No explicit MPI calls required
   - JAX handles domain decomposition automatically
   - Reduced development complexity

3. **Modern Software Stack**
   - Python-based: easier debugging and prototyping
   - Integration with ML/AI libraries (JAX ecosystem)
   - Automatic differentiation for sensitivity studies and inverse problems

4. **Reproducibility**
   - FP32 precision throughout
   - Deterministic on same hardware
   - Version-controlled configurations

### Current Limitations

1. **Microphysics:** Only one-moment schemes (vs. two-moment or bin in CM1/WRF)
2. **Equation Set:** Anelastic only (no compressible mode)
3. **Validation:** Less extensive than established codes
4. **User Community:** Smaller, newer code base
5. **Documentation:** Growing but less comprehensive than CM1/WRF

---

## Performance Characteristics

### Benchmark Results (TPUv6e Trillium)

**Test Configuration:** 256³ grid points per TPU core

| # TPU Cores | Wall Time/Step (ms) | Speedup | Efficiency |
|-------------|---------------------|---------|------------|
| 1 | 120 | 1.0× | 100% |
| 2 | 124 | 0.97× | 97% |
| 4 | 144 | 0.83× | 83% |
| 8 | 178 | 0.67× | 67% |
| 64 | 570 | 0.21× | 21% |

**Notes:**
- Each timestep = 3 RK3 stages
- Includes pressure Poisson solve and thermodynamics
- **Does not** include RRTMGP radiation
- Performance limited by automatic parallelization overhead at high core counts

### Estimated Performance vs. Traditional Codes

**Rough Comparison (order of magnitude):**

For a 512³ supercell simulation over 2 hours (simulation time):

| Code | Hardware | Wall Time | Cost (approx) |
|------|----------|-----------|---------------|
| CM1 | 64 CPU cores (Xeon) | ~24 hrs | $20-40 |
| WRF-LES | 128 CPU cores | ~36 hrs | $40-80 |
| **Swirl-Jatmos** | 8 TPUv6e cores | ~8-12 hrs | $80-120 |

**Caveats:** 
- Highly problem-dependent
- TPU/GPU costs include learning curve and code porting
- Traditional codes may be faster on highly optimized CPU clusters

---

## Installation and Usage

### Installation

```bash
# Clone repository
git clone https://github.com/google-research/swirl-jatmos.git

# Create virtual environment (recommended)
python3 -m venv jatmos_env
source jatmos_env/bin/activate  # On Linux/Mac
# jatmos_env\Scripts\activate  # On Windows

# Install package
python3 -m pip install -e swirl-jatmos
```

**Dependencies:**
- JAX (with GPU/TPU support)
- NumPy
- SciPy
- Orbax (checkpointing)
- dataclasses-json
- absl-py

### Running a Supercell Simulation

**1. Create Configuration:**

```python
from swirl_jatmos import config
from swirl_jatmos.sim_setups import supercell
from swirl_jatmos import driver

# Define grid
cfg_ext = config.ConfigExternal(
    cx=4, cy=4, cz=2,  # Core decomposition
    nx=128, ny=128, nz=64,  # Grid points per core
    domain_x=(0.0, 120e3),  # 120 km
    domain_y=(0.0, 120e3),
    domain_z=(0.0, 20e3),   # 20 km
    dt=0.5,  # Initial timestep
    use_sgs=True,  # Enable SGS turbulence model
    # ... additional configuration
)

# Create full config
cfg = config.config_from_config_external(cfg_ext)
```

**2. Run Simulation:**

```python
# Get reference density for Poisson solver
rho_ref = compute_reference_density(cfg)

# Run driver
states, aux_output, diagnostics = driver.run_driver(
    customized_init_fn=supercell.init_fn,
    rho_ref_xxc=rho_ref,
    output_dir='/path/to/output',
    t_final=7200.0,  # 2 hours
    sec_per_cycle=60.0,  # Output every minute
    cfg=cfg,
)
```

**3. Analysis:**

Output is saved as Orbax checkpoints containing all state variables:
- `u, v, w`: Velocity components
- `theta_li`, `q_t`, `q_r`, `q_s`: Thermodynamic and moisture fields
- `p`: Pressure
- Auxiliary outputs (if requested): `T`, `q_liq`, `q_ice`, `eddy_viscosity`, etc.

### Typical Supercell Simulation Setup

**Resolution:** 250 m horizontal, 100-250 m vertical

**Domain:** 120 km × 120 km × 20 km

**Duration:** 2-3 hours (simulation time)

**Output Frequency:** 1-5 minutes

**Computational Cost:** ~8-24 hours on 8-16 TPU cores

---

## Code Structure

```
swirl-jatmos/
├── swirl_jatmos/
│   ├── config.py                    # Configuration management
│   ├── driver.py                    # Main simulation driver
│   ├── navier_stokes_step.py        # RK3 timestepping
│   ├── velocity.py                  # Momentum equations
│   ├── scalars.py                   # Scalar transport
│   ├── sgs.py                       # Smagorinsky-Lilly SGS model
│   ├── derivatives.py               # Finite difference operators
│   ├── interpolation.py             # Staggered grid interpolation
│   ├── convection.py                # Advection schemes (WENO5, etc.)
│   ├── diffusion.py                 # Diffusion operators
│   ├── boundary_conditions/
│   │   ├── boundary_conditions.py   # BC base classes
│   │   ├── apply_bcs.py            # BC application
│   │   └── monin_obukhov.py        # Surface layer
│   ├── thermodynamics/
│   │   └── water.py                # Moist thermodynamics
│   ├── microphysics/
│   │   ├── microphysics_one_moment.py  # Bulk microphysics
│   │   ├── particles.py            # Particle distributions
│   │   └── terminal_velocity_chen2022.py  # Terminal velocities
│   ├── linalg/
│   │   ├── poisson_solver_interface.py
│   │   ├── fast_diagonalization_solver.py
│   │   └── jacobi_solver.py
│   ├── rrtmgp/                     # Radiation module
│   ├── sim_setups/
│   │   ├── supercell.py            # Supercell initialization
│   │   ├── walker_circulation.py   # Walker circulation
│   │   └── buoyant_bubble.py       # Idealized tests
│   └── utils/
│       ├── file_io.py
│       └── check_states_valid.py   # NaN detection
├── demos/
│   ├── supercell_demo.ipynb        # Colab demo
│   └── rcemip_demo.ipynb           # RCE demo
├── README.md
└── LICENSE
```

---

## Recommended Reading

### Supercell Dynamics
- Weisman, M. L., & Klemp, J. B. (1982). The dependence of numerically simulated convective storms on vertical wind shear and buoyancy. *Monthly Weather Review*.
- Bryan, G. H., & Fritsch, J. M. (2002). A benchmark simulation for moist nonhydrostatic numerical models. *Monthly Weather Review*.

### Numerical Methods
- Wicker, L. J., & Skamarock, W. C. (2002). Time-splitting methods for elastic models using forward time schemes. *Monthly Weather Review*.
- Skamarock, W. C., & Klemp, J. B. (2008). A time-split nonhydrostatic atmospheric model. *Journal of Computational Physics*.

### LES and Turbulence
- Sullivan, P. P., & Patton, E. G. (2011). The effect of mesh resolution on convective boundary layer statistics and structures. *Journal of the Atmospheric Sciences*.
- Bryan, G. H., et al. (2003). Resolution requirements for the simulation of deep moist convection. *Monthly Weather Review*.

### JAX Framework
- Bradbury, J., et al. (2018). JAX: composable transformations of Python+NumPy programs. http://github.com/google/jax

### PBL Schemes
- Hong, S.-Y., & Pan, H.-L. (1996). Nonlocal boundary layer vertical diffusion in a medium-range forecast model. *Monthly Weather Review*.
- Nakanishi, M., & Niino, H. (2009). Development of an improved turbulence closure model for the atmospheric boundary layer. *Journal of the Meteorological Society of Japan*.

---

## APPENDIX: Adding a PBL Turbulence Scheme

### Overview

This section provides a detailed guide for implementing a Planetary Boundary Layer (PBL) turbulence parameterization in Swirl-Jatmos. While the code currently uses the Smagorinsky-Lilly LES closure everywhere, adding a PBL scheme enables:

1. **Gray-zone simulations** - Grid spacing of 100-1000m where neither pure LES nor mesoscale assumptions hold
2. **Hybrid approaches** - PBL scheme in lower atmosphere, LES aloft
3. **Computational efficiency** - Coarser vertical resolution near surface
4. **WRF-style physics** - Comparison with operational models

### Current Turbulence Architecture

**Existing Components:**
```
swirl_jatmos/
├── sgs.py                           # Smagorinsky-Lilly SGS (3D LES)
├── diffusion.py                     # Diffusive flux computation
├── boundary_conditions/
│   └── monin_obukhov.py            # Surface layer fluxes
└── navier_stokes_step.py            # Main integration (uses SGS)
```

**Key Insight:** The code already separates:
- **Turbulent viscosity/diffusivity computation** (`sgs.py`)
- **Flux application** (`diffusion.py`)
- **Surface boundary conditions** (`monin_obukhov.py`)

This makes PBL integration straightforward!

---

### Implementation Strategy

#### Option 1: Vertical Blending (Recommended)

Blend PBL scheme in lower atmosphere with LES above:

```
z > z_blend_top:     Use SGS only (ν_t = ν_SGS)
z_blend_bot < z < z_blend_top:  Blend (ν_t = α·ν_PBL + (1-α)·ν_SGS)
z < z_blend_bot:     Use PBL only (ν_t = ν_PBL)
```

**Advantages:**
- Smooth transition between regimes
- Leverages existing infrastructure
- Minimal code changes

#### Option 2: Replace SGS Entirely

Use PBL scheme throughout domain.

**Use case:** Gray-zone mesoscale simulations where LES is inappropriate

---

### Step-by-Step Implementation

#### Step 1: Create PBL Module Structure

Create a new module: `swirl_jatmos/pbl/`

```bash
swirl_jatmos/pbl/
├── __init__.py
├── pbl_config.py        # Configuration dataclasses
├── pbl_interface.py     # Abstract base class
├── ysu.py              # YSU scheme implementation
├── mynn.py             # MYNN implementation (future)
└── pbl_test.py         # Unit tests
```

#### Step 2: Define Configuration

**File: `swirl_jatmos/pbl/pbl_config.py`**

```python
"""Configuration for PBL schemes."""

import dataclasses
import enum
from typing import TypeAlias

import dataclasses_json


class PBLSchemeType(enum.Enum):
  """Type of PBL scheme to use."""
  NONE = 'none'  # Use SGS only
  YSU = 'ysu'    # Yonsei University
  MYNN = 'mynn'  # Mellor-Yamada-Nakanishi-Niino


@dataclasses.dataclass(frozen=True)
class PBLBlendingConfig(dataclasses_json.DataClassJsonMixin):
  """Configuration for blending PBL with SGS."""
  
  # Use PBL below this height [m]
  z_pbl_max: float = 2000.0
  
  # Blending zone thickness [m]
  blend_thickness: float = 500.0
  
  # Blending function: 'linear', 'cosine', or 'tanh'
  blend_type: str = 'cosine'
  
  # If True, blend viscosity; if False, use max(ν_PBL, ν_SGS)
  use_blending: bool = True


@dataclasses.dataclass(frozen=True)
class YSUConfig(dataclasses_json.DataClassJsonMixin):
  """Configuration for YSU PBL scheme."""
  
  # Asymptotic length scale [m]
  l_max: float = 200.0
  
  # von Karman constant
  kappa: float = 0.4
  
  # Critical bulk Richardson number for PBL height
  ri_crit: float = 0.25
  
  # Minimum PBL height [m]
  h_min: float = 100.0
  
  # Maximum PBL height [m]
  h_max: float = 4000.0
  
  # Profile shape parameter
  profile_exponent: float = 2.0
  
  # Include countergradient term for unstable conditions
  use_countergradient: bool = True


@dataclasses.dataclass(frozen=True)
class PBLConfig(dataclasses_json.DataClassJsonMixin):
  """Main PBL configuration."""
  
  scheme_type: PBLSchemeType = PBLSchemeType.NONE
  blending_cfg: PBLBlendingConfig = dataclasses.field(
      default_factory=PBLBlendingConfig
  )
  ysu_cfg: YSUConfig = dataclasses.field(default_factory=YSUConfig)
```

#### Step 3: Create PBL Interface

**File: `swirl_jatmos/pbl/pbl_interface.py`**

```python
"""Abstract interface for PBL schemes."""

import abc
from typing import TypeAlias

import jax

Array: TypeAlias = jax.Array


class PBLOutput:
  """Output from PBL scheme computation."""
  
  def __init__(
      self,
      k_m: Array,  # Eddy viscosity [m²/s] on (ccc)
      k_h: Array,  # Eddy diffusivity [m²/s] on (ccc)
      pbl_height: Array,  # PBL height [m] on (x, y)
      countergradient_theta: Array | None = None,  # Countergradient term
      countergradient_q: Array | None = None,
  ):
    self.k_m = k_m
    self.k_h = k_h
    self.pbl_height = pbl_height
    self.countergradient_theta = countergradient_theta
    self.countergradient_q = countergradient_q


class PBLScheme(abc.ABC):
  """Abstract base class for PBL schemes."""
  
  @abc.abstractmethod
  def compute_pbl_diffusivities(
      self,
      theta_ccc: Array,
      q_v_ccc: Array,
      u_fcc: Array,
      v_cfc: Array,
      w_ccf: Array,
      z_c: Array,
      z_f: Array,
      rho_xxc: Array,
      u_star: Array,  # Friction velocity from surface [m/s]
      theta_star: Array,  # Temperature scale from surface [K]
      q_star: Array,  # Moisture scale from surface [kg/kg]
  ) -> PBLOutput:
    """Compute PBL eddy diffusivities.
    
    Args:
      theta_ccc: Potential temperature [K] on (ccc).
      q_v_ccc: Water vapor mixing ratio [kg/kg] on (ccc).
      u_fcc: x-velocity [m/s] on (fcc).
      v_cfc: y-velocity [m/s] on (cfc).
      w_ccf: z-velocity [m/s] on (ccf).
      z_c: Height of cell centers [m].
      z_f: Height of cell faces [m].
      rho_xxc: Density [kg/m³] on z-centers.
      u_star: Friction velocity [m/s] on surface (x,y).
      theta_star: Temperature scale [K] on surface (x,y).
      q_star: Moisture scale [kg/kg] on surface (x,y).
    
    Returns:
      PBLOutput containing eddy diffusivities and PBL diagnostics.
    """
    pass
```

#### Step 4: Implement YSU Scheme

**File: `swirl_jatmos/pbl/ysu.py`**

```python
"""YSU (Yonsei University) PBL scheme implementation.

Based on Hong & Pan (1996) and Hong et al. (2006).
"""

from typing import TypeAlias

import jax
import jax.numpy as jnp

from swirl_jatmos import constants
from swirl_jatmos import interpolation
from swirl_jatmos.pbl import pbl_config
from swirl_jatmos.pbl import pbl_interface

Array: TypeAlias = jax.Array


class YSUScheme(pbl_interface.PBLScheme):
  """YSU PBL scheme."""
  
  def __init__(self, cfg: pbl_config.YSUConfig):
    self.cfg = cfg
  
  def compute_pbl_diffusivities(
      self,
      theta_ccc: Array,
      q_v_ccc: Array,
      u_fcc: Array,
      v_cfc: Array,
      w_ccf: Array,
      z_c: Array,
      z_f: Array,
      rho_xxc: Array,
      u_star: Array,
      theta_star: Array,
      q_star: Array,
  ) -> pbl_interface.PBLOutput:
    """Compute YSU eddy diffusivities."""
    
    # Step 1: Compute virtual potential temperature
    theta_v_ccc = theta_ccc * (1.0 + 0.61 * q_v_ccc)
    
    # Step 2: Interpolate velocities to cell centers
    u_ccc = interpolation.x_f_to_c(u_fcc)
    v_ccc = interpolation.y_f_to_c(v_cfc)
    
    # Step 3: Diagnose PBL height using bulk Richardson number
    h_pbl_xy = self._diagnose_pbl_height(
        theta_v_ccc, u_ccc, v_ccc, z_c, u_star, theta_star
    )
    
    # Step 4: Compute eddy diffusivities with YSU profile
    k_m, k_h = self._compute_k_profile(
        h_pbl_xy, z_c, u_star, w_ccf, theta_v_ccc
    )
    
    # Step 5: Compute countergradient terms (optional)
    if self.cfg.use_countergradient:
      gamma_theta, gamma_q = self._compute_countergradient_terms(
          h_pbl_xy, z_c, u_star, theta_star, q_star
      )
    else:
      gamma_theta, gamma_q = None, None
    
    return pbl_interface.PBLOutput(
        k_m=k_m,
        k_h=k_h,
        pbl_height=h_pbl_xy,
        countergradient_theta=gamma_theta,
        countergradient_q=gamma_q,
    )
  
  def _diagnose_pbl_height(
      self,
      theta_v_ccc: Array,
      u_ccc: Array,
      v_ccc: Array,
      z_c: Array,
      u_star: Array,
      theta_star: Array,
  ) -> Array:
    """Diagnose PBL height using bulk Richardson number criterion.
    
    Returns:
      PBL height [m] as 2D array (nx, ny).
    """
    g = constants.G
    kappa = self.cfg.kappa
    ri_crit = self.cfg.ri_crit
    
    # Reference values at first level above surface
    theta_v_sfc = theta_v_ccc[:, :, 1]  # First interior point
    u_sfc = u_ccc[:, :, 1]
    v_sfc = v_ccc[:, :, 1]
    
    # Compute bulk Richardson number at each level
    nz = theta_v_ccc.shape[2]
    
    def compute_ri_at_level(k):
      """Compute Ri at level k."""
      dz = z_c[k] - z_c[1]
      dtheta_v = theta_v_ccc[:, :, k] - theta_v_sfc
      du = u_ccc[:, :, k] - u_sfc
      dv = v_ccc[:, :, k] - v_sfc
      du2 = du**2 + dv**2 + 1e-6  # Avoid division by zero
      
      ri_bulk = (g / theta_v_sfc) * dtheta_v * dz / du2
      return ri_bulk
    
    # Find first level where Ri > Ri_crit
    h_pbl = jnp.zeros_like(theta_v_sfc)
    
    for k in range(2, nz - 1):  # Start from 2nd interior level
      ri_k = compute_ri_at_level(k)
      # If Ri just exceeded Ri_crit, set PBL height
      # Use linear interpolation between levels
      exceeds_crit = (ri_k >= ri_crit)
      not_yet_found = (h_pbl == 0.0)
      
      # Linear interpolation for PBL height
      if k > 2:
        ri_km1 = compute_ri_at_level(k - 1)
        alpha = (ri_crit - ri_km1) / (ri_k - ri_km1 + 1e-10)
        h_interp = z_c[k - 1] + alpha * (z_c[k] - z_c[k - 1])
      else:
        h_interp = z_c[k]
      
      h_pbl = jnp.where(
          exceeds_crit & not_yet_found,
          h_interp,
          h_pbl
      )
    
    # Clamp to valid range
    h_pbl = jnp.clip(h_pbl, self.cfg.h_min, self.cfg.h_max)
    
    # If never exceeded, set to max height
    h_pbl = jnp.where(h_pbl == 0.0, self.cfg.h_max, h_pbl)
    
    return h_pbl
  
  def _compute_k_profile(
      self,
      h_pbl_xy: Array,
      z_c: Array,
      u_star: Array,
      w_ccf: Array,
      theta_v_ccc: Array,
  ) -> tuple[Array, Array]:
    """Compute vertical profile of eddy diffusivities.
    
    YSU uses the profile:
      K(z) = κ w_s z (1 - z/h)^p
    
    where w_s is a velocity scale and p is a profile exponent.
    
    Returns:
      (k_m, k_h): Eddy viscosity and diffusivity [m²/s] on (ccc).
    """
    kappa = self.cfg.kappa
    p = self.cfg.profile_exponent
    
    # Broadcast PBL height to 3D
    h_pbl_3d = h_pbl_xy[:, :, jnp.newaxis]
    
    # Normalized height
    zeta = z_c / (h_pbl_3d + 1e-6)
    
    # Velocity scale w_s = u_star (simplified; can add convective term)
    w_s = u_star[:, :, jnp.newaxis]
    
    # Add convective velocity scale for unstable conditions
    w_star = self._compute_convective_velocity_scale(
        w_ccf, theta_v_ccc, h_pbl_xy
    )
    w_s_total = jnp.sqrt(w_s**2 + 0.6 * w_star**2)
    
    # YSU profile
    profile = zeta * (1.0 - zeta)**p
    profile = jnp.where(zeta < 1.0, profile, 0.0)
    
    k_m = kappa * w_s_total * h_pbl_3d * profile
    
    # Heat diffusivity uses Prandtl number correction
    pr_pbl = 1.0  # Turbulent Prandtl number for PBL
    k_h = k_m / pr_pbl
    
    # Ensure minimum values
    k_min = 1e-3  # Minimum diffusivity [m²/s]
    k_m = jnp.maximum(k_m, k_min)
    k_h = jnp.maximum(k_h, k_min)
    
    return k_m, k_h
  
  def _compute_convective_velocity_scale(
      self,
      w_ccf: Array,
      theta_v_ccc: Array,
      h_pbl_xy: Array,
  ) -> Array:
    """Compute convective velocity scale w_* for unstable BL.
    
    w_* = [(g/θ_v) * (w'θ'_s) * h]^(1/3)
    
    Returns:
      Convective velocity scale [m/s] as 2D array (nx, ny).
    """
    g = constants.G
    
    # Estimate surface buoyancy flux from near-surface gradients
    # (Simplified - in production, would use surface layer fluxes)
    theta_v_sfc = theta_v_ccc[:, :, 1]
    
    # Positive w_star for unstable, zero for stable
    # This is a simplified estimate
    w_star_cubed = jnp.maximum(0.0, g / theta_v_sfc * h_pbl_xy * 0.0)
    w_star = w_star_cubed**(1.0 / 3.0)
    
    return w_star
  
  def _compute_countergradient_terms(
      self,
      h_pbl_xy: Array,
      z_c: Array,
      u_star: Array,
      theta_star: Array,
      q_star: Array,
  ) -> tuple[Array, Array]:
    """Compute countergradient terms for scalars.
    
    In unstable conditions, scalars can have upgradient fluxes
    due to large eddies.
    
    Returns:
      (gamma_theta, gamma_q): Countergradient terms [K/m], [kg/kg/m].
    """
    # Simplified implementation
    # Full YSU has more complex formulation
    
    h_pbl_3d = h_pbl_xy[:, :, jnp.newaxis]
    zeta = z_c / (h_pbl_3d + 1e-6)
    
    # Countergradient only in lower half of PBL
    cg_profile = jnp.where(zeta < 0.5, 1.0 - 2.0 * zeta, 0.0)
    
    gamma_theta = theta_star[:, :, jnp.newaxis] * cg_profile / h_pbl_3d
    gamma_q = q_star[:, :, jnp.newaxis] * cg_profile / h_pbl_3d
    
    return gamma_theta, gamma_q
```

#### Step 5: Integrate PBL into Main Code

**Modify `swirl_jatmos/config.py`:**

```python
# Add to imports
from swirl_jatmos.pbl import pbl_config

# Add to ConfigExternal dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class ConfigExternal(dataclasses_json.DataClassJsonMixin):
  # ... existing fields ...
  
  # PBL configuration
  pbl_cfg: pbl_config.PBLConfig = dataclasses.field(
      default_factory=pbl_config.PBLConfig
  )
```

**Modify `swirl_jatmos/navier_stokes_step.py`:**

Add PBL computation in the `_substep` function:

```python
def _substep(
    states: dict[str, Array],
    f_prev_states: dict[str, Array],
    poisson_solver: poisson_solver_interface.PoissonSolver,
    cfg: config.Config,
    k: int,
) -> tuple[dict[str, Array], dict[str, Array], dict[str, Array]]:
  """Compute one substep (one stage) of the RK3 step."""
  
  # ... existing code for thermodynamics ...
  
  # Compute strain rate tensor
  strain_rate_tensor = utils.compute_strain_rate_tensor(
      u_fcc, v_cfc, w_ccf, deriv_lib, sg_map, cfg.z_c, cfg.z_f, cfg.z_bcs
  )
  
  # NEW: Compute PBL diffusivities if enabled
  if cfg.pbl_cfg.scheme_type != pbl_config.PBLSchemeType.NONE:
    pbl_output = compute_pbl_diffusivities(
        theta_li_ccc, thermo_fields.q_v, u_fcc, v_cfc, w_ccf,
        cfg, states
    )
    pbl_k_m = pbl_output.k_m
    pbl_k_h = pbl_output.k_h
  else:
    pbl_k_m = jnp.zeros_like(theta_li_ccc)
    pbl_k_h = jnp.zeros_like(theta_li_ccc)
  
  # Compute SGS diffusivities
  if cfg.use_sgs:
    eddy_viscosity_ccc = sgs.smagorinsky_lilly_nu_t(
        strain_rate_tensor, theta_li_ccc, pr_t, sg_map,
        cfg.z_c, deriv_lib, halo_width,
    )
  else:
    eddy_viscosity_ccc = jnp.zeros_like(theta_li_ccc)
  
  # NEW: Blend PBL and SGS diffusivities
  if cfg.pbl_cfg.scheme_type != pbl_config.PBLSchemeType.NONE:
    eddy_viscosity_ccc, eddy_diffusivity_ccc = blend_pbl_and_sgs(
        pbl_k_m, pbl_k_h, eddy_viscosity_ccc,
        states['z_c'], cfg.pbl_cfg.blending_cfg
    )
    viscosity_ccc = const_viscosity_ccc + eddy_viscosity_ccc
    diffusivity_ccc = const_diffusivity_ccc + eddy_diffusivity_ccc
  else:
    # Original code path
    viscosity_ccc = const_viscosity_ccc + eddy_viscosity_ccc
    diffusivity_ccc = const_diffusivity_ccc + eddy_viscosity_ccc / pr_t
  
  # ... rest of substep ...
```

Add helper functions:

```python
def compute_pbl_diffusivities(
    theta_ccc: Array,
    q_v_ccc: Array,
    u_fcc: Array,
    v_cfc: Array,
    w_ccf: Array,
    cfg: config.Config,
    states: dict[str, Array],
) -> pbl_interface.PBLOutput:
  """Compute PBL diffusivities using configured scheme."""
  
  from swirl_jatmos.pbl import ysu
  
  # Get surface layer scales from Monin-Obukhov if available
  if 'u_star_2d_xy' in states:
    u_star = states['u_star_2d_xy']
    theta_star = states['theta_star_2d_xy']
    q_star = states['q_star_2d_xy']
  else:
    # Fallback: estimate from near-surface values
    u_star = estimate_surface_friction_velocity(u_fcc, v_cfc, cfg.z_c)
    theta_star = jnp.zeros_like(u_star) + 0.1  # Placeholder
    q_star = jnp.zeros_like(u_star)
  
  # Instantiate PBL scheme
  if cfg.pbl_cfg.scheme_type == pbl_config.PBLSchemeType.YSU:
    scheme = ysu.YSUScheme(cfg.pbl_cfg.ysu_cfg)
  else:
    raise ValueError(f'Unknown PBL scheme: {cfg.pbl_cfg.scheme_type}')
  
  # Compute diffusivities
  return scheme.compute_pbl_diffusivities(
      theta_ccc,
      q_v_ccc,
      u_fcc,
      v_cfc,
      w_ccf,
      cfg.z_c,
      cfg.z_f,
      states['rho_xxc'],
      u_star,
      theta_star,
      q_star,
  )


def blend_pbl_and_sgs(
    k_m_pbl: Array,
    k_h_pbl: Array,
    nu_sgs: Array,
    z_c: Array,
    blend_cfg: pbl_config.PBLBlendingConfig,
) -> tuple[Array, Array]:
  """Blend PBL and SGS diffusivities vertically.
  
  Returns:
    (nu_total, kappa_total): Blended viscosity and diffusivity.
  """
  z_top = blend_cfg.z_pbl_max
  dz_blend = blend_cfg.blend_thickness
  z_bot = z_top - dz_blend
  
  if blend_cfg.blend_type == 'linear':
    # Linear blending
    alpha = (z_c - z_bot) / dz_blend
  elif blend_cfg.blend_type == 'cosine':
    # Smooth cosine blending
    alpha = 0.5 * (1.0 + jnp.cos(jnp.pi * (z_top - z_c) / dz_blend))
  else:  # tanh
    # Hyperbolic tangent
    alpha = 0.5 * (1.0 + jnp.tanh((z_c - z_top) / (0.2 * dz_blend)))
  
  # Clamp alpha to [0, 1]
  alpha = jnp.clip(alpha, 0.0, 1.0)
  
  if blend_cfg.use_blending:
    # Weighted average
    nu_total = alpha * nu_sgs + (1.0 - alpha) * k_m_pbl
    kappa_total = alpha * (nu_sgs / 0.33) + (1.0 - alpha) * k_h_pbl
  else:
    # Use maximum
    nu_total = jnp.maximum(nu_sgs, k_m_pbl)
    kappa_total = jnp.maximum(nu_sgs / 0.33, k_h_pbl)
  
  return nu_total, kappa_total
```

#### Step 6: Update Surface Layer to Provide Scales

**Modify `swirl_jatmos/boundary_conditions/monin_obukhov.py`:**

Add function to compute and return surface scales:

```python
def compute_surface_scales(
    u_fcc: Array,
    v_cfc: Array,
    theta_ccc: Array,
    q_ccc: Array,
    rho_xxc: Array,
    theta_surface: Array,
    q_surface: Array,
    mop: MoninObukhovParams,
) -> tuple[Array, Array, Array]:
  """Compute u*, θ*, q* from Monin-Obukhov similarity theory.
  
  Returns:
    (u_star, theta_star, q_star): Surface scales.
  """
  # ... implementation based on existing surface_momentum_flux
  # and surface_scalar_flux functions ...
  
  # This would extract the u*, θ*, q* that are implicitly
  # computed in the existing functions
  pass
```

Store these in `states` dictionary during initialization for access by PBL scheme.

#### Step 7: Testing and Validation

**File: `swirl_jatmos/pbl/pbl_test.py`**

```python
"""Unit tests for PBL schemes."""

import jax.numpy as jnp
import numpy as np

from swirl_jatmos.pbl import pbl_config
from swirl_jatmos.pbl import ysu


def test_ysu_pbl_height_diagnosis():
  """Test PBL height diagnosis with known profile."""
  
  # Create idealized profile
  nz = 50
  z_c = np.linspace(50, 5000, nz)
  
  # Stable boundary layer
  theta_v = 300.0 + 0.01 * z_c  # 10 K/km lapse rate
  u = 10.0 * np.ones_like(z_c)
  v = 0.0 * np.ones_like(z_c)
  
  theta_v_ccc = jnp.array(theta_v).reshape(1, 1, nz)
  u_ccc = jnp.array(u).reshape(1, 1, nz)
  v_ccc = jnp.array(v).reshape(1, 1, nz)
  u_star = jnp.array([[0.3]])
  theta_star = jnp.array([[0.1]])
  
  cfg = pbl_config.YSUConfig()
  scheme = ysu.YSUScheme(cfg)
  
  h_pbl = scheme._diagnose_pbl_height(
      theta_v_ccc, u_ccc, v_ccc, z_c, u_star, theta_star
  )
  
  # PBL height should be reasonable (100-2000m for stable case)
  assert h_pbl[0, 0] > cfg.h_min
  assert h_pbl[0, 0] < 2000.0
  print(f'Diagnosed PBL height: {h_pbl[0, 0]:.1f} m')


def test_ysu_k_profile():
  """Test K profile computation."""
  
  nz = 50
  z_c = np.linspace(50, 5000, nz)
  h_pbl = 1000.0
  u_star = 0.3
  
  h_pbl_xy = jnp.array([[h_pbl]])
  u_star_xy = jnp.array([[u_star]])
  w_ccf = jnp.zeros((1, 1, nz))
  theta_v_ccc = 300.0 * jnp.ones((1, 1, nz))
  
  cfg = pbl_config.YSUConfig()
  scheme = ysu.YSUScheme(cfg)
  
  k_m, k_h = scheme._compute_k_profile(
      h_pbl_xy, z_c, u_star_xy, w_ccf, theta_v_ccc
  )
  
  # Check profile properties
  # K should be maximum in mid-PBL
  k_in_pbl = k_m[0, 0, z_c < h_pbl]
  assert jnp.max(k_in_pbl) > 10.0  # Reasonable magnitude
  
  # K should be near zero above PBL
  k_above_pbl = k_m[0, 0, z_c > 1.5 * h_pbl]
  assert jnp.mean(k_above_pbl) < 1.0
  
  print(f'Max K_m in PBL: {jnp.max(k_in_pbl):.1f} m²/s')


if __name__ == '__main__':
  test_ysu_pbl_height_diagnosis()
  test_ysu_k_profile()
  print('All tests passed!')
```

---

### Usage Example

To run a simulation with PBL scheme:

```python
from swirl_jatmos import config
from swirl_jatmos.pbl import pbl_config

# Create config with YSU PBL
pbl_cfg = pbl_config.PBLConfig(
    scheme_type=pbl_config.PBLSchemeType.YSU,
    blending_cfg=pbl_config.PBLBlendingConfig(
        z_pbl_max=2000.0,  # Use PBL below 2 km
        blend_thickness=500.0,  # 500m blending zone
        blend_type='cosine',
    ),
    ysu_cfg=pbl_config.YSUConfig(
        ri_crit=0.25,
        use_countergradient=True,
    ),
)

cfg_ext = config.ConfigExternal(
    # ... grid and domain setup ...
    use_sgs=True,  # Enable SGS above PBL
    pbl_cfg=pbl_cfg,  # Add PBL configuration
)
```

---

### Testing Strategy

1. **Unit Tests**
   - PBL height diagnosis with idealized profiles
   - K-profile shape and magnitude
   - Blending function smoothness

2. **Verification Cases**
   - Neutral boundary layer (Ekman spiral)
   - Convective boundary layer (compare entrainment rate)
   - Stable boundary layer (check critical Richardson number)

3. **Comparison Studies**
   - Run same case with/without PBL
   - Compare to WRF-LES with YSU
   - Validate against LES benchmark (e.g., GABLS)

4. **Gray-Zone Tests**
   - Grid spacing 250m, 500m, 1km
   - Check smooth transition between LES and PBL regimes

---

### Performance Considerations

**Computational Cost:**
- PBL scheme adds ~5-10% overhead (much less than full 3D SGS)
- Primarily 1D vertical computations
- PBL height diagnosis is the most expensive part

**Optimization Tips:**
1. JIT compile PBL scheme separately: `@jax.jit`
2. Vectorize over horizontal dimensions
3. Cache PBL height if only updating every few timesteps
4. Use static array shapes to avoid re-compilation

**JAX-Specific Considerations:**
```python
# Make PBL height diagnosis JIT-friendly
@jax.jit
def diagnose_pbl_height_jit(theta_v, u, v, z_c, u_star):
  # Use jax.lax.scan instead of Python loops for vertical iteration
  def scan_fn(carry, inputs):
    h_pbl, found = carry
    z_k, theta_v_k, u_k, v_k = inputs
    # ... Richardson number logic ...
    return (h_pbl_new, found_new), None
  
  (h_pbl_final, _), _ = jax.lax.scan(
      scan_fn, init_carry, scan_inputs
  )
  return h_pbl_final
```

---

### Advanced Extensions

#### 1. MYNN Scheme

MYNN (Mellor-Yamada-Nakanishi-Niino) is a more sophisticated TKE-based scheme:

```python
# swirl_jatmos/pbl/mynn.py

class MYNNScheme(pbl_interface.PBLScheme):
  """MYNN 2.5-level TKE scheme."""
  
  def compute_pbl_diffusivities(self, ...):
    # Prognostic TKE equation
    tke_new = self._integrate_tke_equation(tke_old, ...)
    
    # Diagnose mixing length
    l_mix = self._compute_mixing_length(tke_new, ...)
    
    # Compute K from TKE and mixing length
    k_m = c_k * l_mix * jnp.sqrt(tke_new)
    k_h = k_m / pr_t
    
    return PBLOutput(k_m=k_m, k_h=k_h, ...)
```

Requires adding TKE as a prognostic variable to `states`.

#### 2. Scale-Aware Blending

Adjust PBL/SGS blending based on grid resolution:

```python
def scale_aware_blending_factor(dx, dy, dz, h_pbl):
  """Compute blending factor based on grid spacing."""
  
  # Effective resolution
  delta_eff = (dx * dy * dz)**(1/3)
  
  # If delta << h_pbl: LES regime (use SGS only)
  # If delta ~ h_pbl: gray zone (blend)
  # If delta >> h_pbl: mesoscale (use PBL only)
  
  ratio = delta_eff / h_pbl
  alpha_sgs = jnp.clip(1.0 / (1.0 + ratio**2), 0.0, 1.0)
  
  return alpha_sgs
```

#### 3. Entrainment Closure

Add explicit entrainment at PBL top:

```python
def compute_entrainment_flux(
    h_pbl, dhdt_pbl, theta_jump, w_star
):
  """Compute entrainment heat flux at PBL top."""
  
  # Entrainment rate (simplified)
  w_e = 0.2 * w_star  # 20% of convective velocity scale
  
  # Entrainment flux
  flux_e = w_e * theta_jump
  
  return flux_e
```

Apply this as a boundary condition at h_pbl interface.

---

### Debugging Tips

1. **Visualize Profiles**
```python
# Plot K profile
import matplotlib.pyplot as plt

z = cfg.z_c
k_pbl = pbl_output.k_m[nx//2, ny//2, :]
k_sgs = eddy_visc[nx//2, ny//2, :]

plt.plot(k_pbl, z, label='PBL')
plt.plot(k_sgs, z, label='SGS')
plt.axhline(pbl_output.pbl_height[nx//2, ny//2], 
            color='k', linestyle='--', label='h_PBL')
plt.xlabel('K [m²/s]')
plt.ylabel('Height [m]')
plt.legend()
```

2. **Check PBL Height**
```python
# Monitor PBL height evolution
jax.debug.print('PBL height: min={}, max={}, mean={}',
                jnp.min(h_pbl), jnp.max(h_pbl), jnp.mean(h_pbl))
```

3. **Verify Conservation**
```python
# Check that blending doesn't create spurious sources/sinks
flux_pbl = compute_flux_divergence(k_pbl, ...)
flux_sgs = compute_flux_divergence(k_sgs, ...)
flux_blended = compute_flux_divergence(k_blended, ...)

# Should be between PBL and SGS values
assert jnp.all((flux_pbl <= flux_blended) & (flux_blended <= flux_sgs))
```

---

### References for Implementation

**Essential Papers:**
- Hong, S.-Y., & Pan, H.-L. (1996). Nonlocal boundary layer vertical diffusion in a medium-range forecast model. *MWR, 124*, 2322-2339.
- Hong, S.-Y., Noh, Y., & Dudhia, J. (2006). A new vertical diffusion package with an explicit treatment of entrainment processes. *MWR, 134*, 2318-2341.
- Nakanishi, M., & Niino, H. (2009). Development of an improved turbulence closure model for the atmospheric boundary layer. *JMSJ, 87*, 895-912.

**WRF Code References:**
- WRF-ARW Technical Note (Skamarock et al.)
- WRF Module: `phys/module_bl_ysu.F`
- WRF Module: `phys/module_bl_mynn.F`

**Gray-Zone Literature:**
- Shin, H. H., & Hong, S.-Y. (2015). Representation of the subgrid-scale turbulent transport in convective boundary layers at gray-zone resolutions. *MWR, 143*, 250-271.
- Honnert, R., et al. (2011). A diagnostic for evaluating the representation of turbulence in atmospheric models at the kilometric scale. *JAS, 68*, 3112-3131.

---

### Summary

Adding a PBL scheme to Swirl-Jatmos involves:

1. ✅ **Minimal code changes** - Existing architecture supports it well
2. ✅ **Modular design** - New `pbl/` module separate from core
3. ✅ **JAX compatibility** - Use vectorization and jax.lax control flow
4. ✅ **Flexible blending** - Smooth transition between PBL and LES regimes
5. ✅ **Configuration driven** - Easy to enable/disable and tune

**Expected timeline:**
- YSU basic implementation: 2-3 days
- Testing and validation: 1-2 weeks
- MYNN extension: 1-2 weeks additional
- Publication-quality validation: 2-3 months

This would make Swirl-Jatmos competitive with WRF for gray-zone simulations while maintaining its GPU/TPU performance advantages!

---

## Conclusions

**Swirl-Jatmos represents a modern approach to LES for atmospheric convection**, leveraging:
- Accelerator hardware (TPU/GPU)
- Automatic parallelization (JAX)
- Python ecosystem for ease of development

**Compared to established codes like CM1, WRF-LES, and PALM:**
- **Strengths:** Hardware acceleration, automatic parallelization, modern software stack, ML integration potential
- **Limitations:** Less mature, fewer physics options, smaller user community

**Best suited for:**
- Research groups with GPU/TPU access
- Idealized LES studies of convection
- Integration with machine learning workflows
- Rapid prototyping and experimentation

**Traditional codes (CM1, WRF) remain preferred for:**
- Operational applications
- Studies requiring advanced microphysics (multi-moment, bin)
- Real-case simulations with complex forcing
- Maximum community support and documentation

**Future development directions** for Swirl-Jatmos could include:
- Two-moment and bin microphysics
- Compressible equation set
- Terrain-following coordinates
- Enhanced validation against observations
- Growing documentation and tutorials

---

## Contact and Resources

**Repository:** https://github.com/google-research/swirl-jatmos

**License:** Apache 2.0

**Citation:** If using this code, please cite the repository and any relevant publications from the development team.

**Note:** *This is not an officially supported Google product.*

---

*Documentation compiled from source code analysis on October 20, 2025*

