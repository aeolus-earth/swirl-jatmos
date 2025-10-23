# Guide: Adding Mesoscale Weather Model Boundary Conditions to Swirl-Jatmos

## Executive Summary

This guide outlines how to integrate lateral boundary conditions from a mesoscale weather model (e.g., WRF, WRF-ARW, MPAS) into swirl-jatmos, a JAX-based atmospheric LES model. Currently, swirl-jatmos only supports **periodic boundary conditions** in the horizontal (x, y) directions. This document provides a detailed analysis and implementation strategy for adding **open/specified boundary conditions** from mesoscale model output.

---

## Current State Analysis

### 1. Existing Boundary Condition Architecture

**Horizontal Boundaries (x, y):**
- **Type:** Periodic only (hardcoded via `jnp.roll` operations)
- **Implementation:** Located in `kernel_ops.py` - all spatial operators use JAX roll operations
- **Location:** Used throughout interpolation, derivatives, convection modules
- **Key limitation:** No infrastructure for non-periodic lateral boundaries

**Vertical Boundaries (z):**
- **Types supported:**
  - `'no_flux'`: Free-slip walls (w=0, no diffusive flux)
  - `'monin_obukhov'`: Surface layer similarity theory
- **Configuration:** Via `boundary_conditions.ZBoundaryConditions` dataclass
- **Implementation:** In `swirl_jatmos/boundary_conditions/` directory
- **Application:** Explicit flux enforcement at boundaries

### 2. Key Code Locations

```
swirl_jatmos/
├── boundary_conditions/
│   ├── boundary_conditions.py    # BC configuration classes
│   ├── apply_bcs.py               # BC application functions
│   └── monin_obukhov.py          # Surface layer BC
├── kernel_ops.py                  # Periodic roll operations
├── interpolation.py               # Stencil operations (uses roll)
├── derivatives.py                 # Spatial derivatives
├── convection.py                  # Convective fluxes
├── navier_stokes_step.py         # Main time stepping
├── config.py                      # Configuration system
└── sim_initializer.py             # Initialization & grid setup
```

### 3. Grid Structure

- **Type:** Arakawa C-grid (staggered)
- **Prognostic variables:**
  - `u` (x-velocity): at x-faces (i+1/2, j, k)
  - `v` (y-velocity): at y-faces (i, j+1/2, k)
  - `w` (z-velocity): at z-faces (i, j, k+1/2)
  - `theta_li` (liquid-ice potential temp): at cell centers
  - `q_t` (total specific humidity): at cell centers
  - `q_r`, `q_s` (rain, snow): at cell centers
- **Grid spacing:** Supports uniform or stretched grids in each direction
- **Halo width:** 1 (hardcoded)

### 4. Mesoscale Model Data Requirements

**Typical WRF output that could drive swirl-jatmos:**

| Variable | WRF Name | Swirl-Jatmos Equivalent | Grid Location |
|----------|----------|-------------------------|---------------|
| U wind | U | u | x-face |
| V wind | V | v | y-face |
| W wind | W | w | z-face |
| Potential temp | T (+ base) | theta_li | center |
| Water vapor mixing | QVAPOR | q_t | center |
| Rain mixing | QRAIN | q_r | center |
| Snow mixing | QSNOW | q_s | center |
| Pressure | P (+ base) | p_ref_xxc | center |
| Density | (calculated) | rho_xxc | center |

---

## Implementation Strategy

### Phase 1: Lateral Boundary Condition Framework

#### 1.1 Extend Boundary Condition Types

**Create new BC type enum in `boundary_conditions.py`:**

```python
@dataclasses.dataclass(frozen=True, kw_only=True)
class LateralBC(dataclasses_json.DataClassJsonMixin):
  """Parameters for lateral (x, y) boundary conditions."""
  
  # BC types: 'periodic', 'specified', 'relaxation', 'open_radiation'
  bc_type: Literal['periodic', 'specified', 'relaxation', 'open_radiation'] = 'periodic'
  
  # For specified BCs: path to mesoscale model data
  forcing_data_path: str | None = None
  
  # Relaxation zone width (number of grid points)
  relaxation_width: int = 10
  
  # Relaxation timescale (seconds)
  relaxation_timescale: float = 600.0
  
  # Update frequency for boundary data (timesteps)
  update_interval: int = 10


@dataclasses.dataclass(frozen=True, kw_only=True)
class BoundaryConditions(dataclasses_json.DataClassJsonMixin):
  """Complete boundary condition specification."""
  
  x_west: LateralBC = LateralBC()
  x_east: LateralBC = LateralBC()
  y_south: LateralBC = LateralBC()
  y_north: LateralBC = LateralBC()
  z_bottom: boundary_conditions.ZBC = boundary_conditions.ZBC()
  z_top: boundary_conditions.ZBC = boundary_conditions.ZBC()
```

#### 1.2 Boundary Data Loader

**Create `swirl_jatmos/boundary_conditions/mesoscale_forcing.py`:**

```python
"""Module for loading and interpolating mesoscale model boundary data."""

import dataclasses
from typing import TypeAlias
import jax
import jax.numpy as jnp
import netCDF4 as nc
from swirl_jatmos import config
from swirl_jatmos import interpolation

Array: TypeAlias = jax.Array


@dataclasses.dataclass
class BoundaryData:
  """Container for boundary forcing data at a single time."""
  
  # West boundary (yz plane at x=0)
  west_u: Array | None = None  # shape: (1, ny, nz)
  west_v: Array | None = None
  west_w: Array | None = None
  west_theta_li: Array | None = None
  west_q_t: Array | None = None
  west_q_r: Array | None = None
  west_q_s: Array | None = None
  
  # East boundary (yz plane at x=nx-1)
  east_u: Array | None = None
  east_v: Array | None = None
  # ... similar for east, north, south
  
  # Time stamp (seconds since simulation start)
  time: float = 0.0


class MesoscaleForcingReader:
  """Reads and interpolates mesoscale model output for LES boundaries."""
  
  def __init__(
      self,
      wrf_file_path: str,
      les_config: config.Config,
      les_domain_center_lat: float,
      les_domain_center_lon: float,
  ):
    """Initialize the forcing reader.
    
    Args:
      wrf_file_path: Path to WRF output file (wrfout_d0X_YYYY-MM-DD_HH:MM:SS)
      les_config: Swirl-jatmos configuration
      les_domain_center_lat: Latitude of LES domain center
      les_domain_center_lon: Longitude of LES domain center
    """
    self.wrf_file = nc.Dataset(wrf_file_path, 'r')
    self.les_config = les_config
    self.center_lat = les_domain_center_lat
    self.center_lon = les_domain_center_lon
    
    # Find WRF grid indices corresponding to LES domain
    self._map_wrf_to_les_domain()
    
  def _map_wrf_to_les_domain(self):
    """Map LES domain to WRF grid indices."""
    # Read WRF grid
    wrf_lat = self.wrf_file.variables['XLAT'][0, :, :]  # (south_north, west_east)
    wrf_lon = self.wrf_file.variables['XLONG'][0, :, :]
    
    # Find WRF indices closest to LES domain corners
    # This is simplified - in practice, need proper map projection handling
    # and rotation for domains not aligned with WRF grid
    
    # Example: find center point
    distances = (wrf_lat - self.center_lat)**2 + (wrf_lon - self.center_lon)**2
    center_j, center_i = jnp.unravel_index(jnp.argmin(distances), distances.shape)
    
    # Calculate domain extent in grid points
    les_lx = self.les_config.domain_x[1] - self.les_config.domain_x[0]
    les_ly = self.les_config.domain_y[1] - self.les_config.domain_y[0]
    
    # Estimate WRF grid spacing (assume uniform for simplicity)
    # In reality, need to use map factors
    dx_wrf = 3000.0  # Example: 3 km WRF grid spacing
    
    # Calculate index range
    nx_wrf = int(les_lx / dx_wrf)
    ny_wrf = int(les_ly / dx_wrf)
    
    self.wrf_i_min = center_i - nx_wrf // 2
    self.wrf_i_max = center_i + nx_wrf // 2
    self.wrf_j_min = center_j - ny_wrf // 2
    self.wrf_j_max = center_j + ny_wrf // 2
    
  def read_boundary_at_time(self, time_idx: int) -> BoundaryData:
    """Read WRF data and extract boundary values.
    
    Args:
      time_idx: WRF time index
      
    Returns:
      BoundaryData object with interpolated boundary values
    """
    # Read WRF variables at time_idx
    # Note: WRF uses staggered grid similar to swirl-jatmos
    u_wrf = self.wrf_file.variables['U'][time_idx, :, :, :]  # (z, y, x_stag)
    v_wrf = self.wrf_file.variables['V'][time_idx, :, :, :]  # (z, y_stag, x)
    w_wrf = self.wrf_file.variables['W'][time_idx, :, :, :]  # (z_stag, y, x)
    
    # Potential temperature (need to add base state)
    t_wrf = self.wrf_file.variables['T'][time_idx, :, :, :]  # perturbation
    t_base = self.wrf_file.variables['T00'][time_idx]
    theta_wrf = t_wrf + t_base
    
    # Moisture
    qv_wrf = self.wrf_file.variables['QVAPOR'][time_idx, :, :, :]
    qr_wrf = self.wrf_file.variables['QRAIN'][time_idx, :, :, :]
    
    # Extract boundaries and interpolate to LES grid
    # West boundary (western edge of LES domain)
    west_u = self._interpolate_to_les_grid(
        u_wrf[:, self.wrf_j_min:self.wrf_j_max, self.wrf_i_min],
        axis='west',
        variable_type='u'
    )
    
    # Similar for other boundaries...
    
    return BoundaryData(
        west_u=west_u,
        # ... other fields
        time=float(self.wrf_file.variables['XTIME'][time_idx])
    )
  
  def _interpolate_to_les_grid(
      self,
      wrf_data: jnp.ndarray,
      axis: str,
      variable_type: str
  ) -> Array:
    """Interpolate WRF boundary data to LES grid.
    
    Args:
      wrf_data: WRF data slice (2D: y-z or x-z)
      axis: 'west', 'east', 'south', or 'north'
      variable_type: 'u', 'v', 'w', 'theta', etc.
      
    Returns:
      Interpolated data on LES grid
    """
    # Vertical interpolation: WRF uses terrain-following coordinates
    # Need to interpolate from WRF z-levels to LES z-levels
    
    # Horizontal interpolation along boundary
    
    # This is a placeholder - actual implementation requires:
    # 1. Vertical coordinate transformation (WRF eta -> LES z)
    # 2. Horizontal interpolation (WRF grid spacing -> LES grid spacing)
    # 3. Handling of staggered grid differences
    # 4. Map projection transformations if domains not aligned
    
    # For now, return dummy data
    if axis in ['west', 'east']:
      ny_les = len(self.les_config.y_c)
      nz_les = len(self.les_config.z_c)
      return jnp.zeros((1, ny_les, nz_les))
    else:  # north/south
      nx_les = len(self.les_config.x_c)
      nz_les = len(self.les_config.z_c)
      return jnp.zeros((nx_les, 1, nz_les))
```

#### 1.3 Boundary Condition Application

**Create `swirl_jatmos/boundary_conditions/lateral_bcs.py`:**

```python
"""Module for applying lateral boundary conditions."""

from typing import TypeAlias, Literal
import jax
import jax.numpy as jnp
from swirl_jatmos.boundary_conditions import mesoscale_forcing

Array: TypeAlias = jax.Array
BoundaryData: TypeAlias = mesoscale_forcing.BoundaryData


def apply_relaxation_zone(
    field_interior: Array,
    field_boundary: Array,
    relaxation_width: int,
    relaxation_timescale: float,
    dt: float,
    axis: Literal['x', 'y'],
    side: Literal['min', 'max'],
) -> Array:
  """Apply relaxation zone to blend interior solution with boundary data.
  
  Uses Newtonian relaxation (nudging):
    dφ/dt = -(φ - φ_boundary) / τ * weight(distance)
  
  where weight varies from 1 at boundary to 0 at edge of relaxation zone.
  
  Args:
    field_interior: Interior field values (full 3D array)
    field_boundary: Boundary field values (2D slice)
    relaxation_width: Width of relaxation zone in grid points
    relaxation_timescale: Relaxation timescale τ (seconds)
    dt: Time step (seconds)
    axis: 'x' or 'y'
    side: 'min' (west/south) or 'max' (east/north)
    
  Returns:
    Updated field with relaxation applied
  """
  # Create weight function: linear ramp from 1 to 0
  # weight = 1 at boundary, 0 at relaxation_width distance
  weights = jnp.linspace(1.0, 0.0, relaxation_width)
  
  # Reshape weights for broadcasting
  if axis == 'x':
    if side == 'min':  # west
      weights = weights.reshape(relaxation_width, 1, 1)
      zone_slice = jnp.s_[:relaxation_width, :, :]
      boundary_expanded = field_boundary[0, :, :].reshape(1, -1, field_boundary.shape[2])
    else:  # east
      weights = weights[::-1].reshape(relaxation_width, 1, 1)
      zone_slice = jnp.s_[-relaxation_width:, :, :]
      boundary_expanded = field_boundary[-1, :, :].reshape(1, -1, field_boundary.shape[2])
  else:  # axis == 'y'
    if side == 'min':  # south
      weights = weights.reshape(1, relaxation_width, 1)
      zone_slice = jnp.s_[:, :relaxation_width, :]
      boundary_expanded = field_boundary[:, 0, :].reshape(-1, 1, field_boundary.shape[2])
    else:  # north
      weights = weights[::-1].reshape(1, relaxation_width, 1)
      zone_slice = jnp.s_[:, -relaxation_width:, :]
      boundary_expanded = field_boundary[:, -1, :].reshape(-1, 1, field_boundary.shape[2])
  
  # Apply relaxation: φ^(n+1) = φ^n - dt/τ * weight * (φ^n - φ_boundary)
  relaxation_tendency = -weights / relaxation_timescale * (
      field_interior[zone_slice] - boundary_expanded
  )
  
  updated_field = field_interior.at[zone_slice].add(relaxation_tendency * dt)
  
  return updated_field


def apply_specified_bc(
    field: Array,
    boundary_value: Array,
    axis: Literal[0, 1],  # 0=x, 1=y
    side: Literal['min', 'max'],
) -> Array:
  """Apply specified (Dirichlet) boundary condition.
  
  Args:
    field: Full 3D field
    boundary_value: Boundary values (2D array)
    axis: 0 for x, 1 for y
    side: 'min' or 'max'
    
  Returns:
    Field with boundary values applied
  """
  if axis == 0:  # x direction
    if side == 'min':
      field = field.at[0, :, :].set(boundary_value.squeeze())
    else:
      field = field.at[-1, :, :].set(boundary_value.squeeze())
  else:  # y direction
    if side == 'min':
      field = field.at[:, 0, :].set(boundary_value.squeeze())
    else:
      field = field.at[:, -1, :].set(boundary_value.squeeze())
  
  return field


def update_lateral_boundaries(
    states: dict[str, Array],
    boundary_data: BoundaryData,
    lateral_bcs: dict,
    dt: float,
) -> dict[str, Array]:
  """Update all lateral boundaries for prognostic variables.
  
  Args:
    states: State dictionary containing u, v, w, theta_li, q_t, q_r, q_s
    boundary_data: BoundaryData object with forcing values
    lateral_bcs: Dictionary of LateralBC configuration for each boundary
    dt: Timestep
    
  Returns:
    Updated states dictionary
  """
  updated_states = states.copy()
  
  # Process each boundary
  for boundary_name, bc_config in lateral_bcs.items():
    if bc_config.bc_type == 'periodic':
      # No action needed - kernel_ops already handles this via roll
      continue
      
    elif bc_config.bc_type == 'relaxation':
      # Apply relaxation zones
      # West boundary example
      if boundary_name == 'x_west':
        for var in ['u', 'v', 'w', 'theta_li', 'q_t', 'q_r', 'q_s']:
          if var in updated_states and hasattr(boundary_data, f'west_{var}'):
            boundary_val = getattr(boundary_data, f'west_{var}')
            if boundary_val is not None:
              updated_states[var] = apply_relaxation_zone(
                  updated_states[var],
                  boundary_val,
                  bc_config.relaxation_width,
                  bc_config.relaxation_timescale,
                  dt,
                  axis='x',
                  side='min',
              )
      
      # Similar for east, north, south boundaries...
      
    elif bc_config.bc_type == 'specified':
      # Apply Dirichlet BC
      # Implementation similar to relaxation but with apply_specified_bc
      pass
  
  return updated_states
```

### Phase 2: Modify Kernel Operations

Currently, all spatial operations use `jnp.roll` which assumes periodicity. Need to:

#### 2.1 Create Conditional Kernel Operations

**Modify `kernel_ops.py` to support non-periodic operations:**

```python
def forward_sum_nonperiodic(
    f: Array,
    dim: Literal[0, 1, 2],
    fill_value: float = 0.0
) -> Array:
  """Forward sum with non-periodic boundary handling.
  
  For periodic: uses roll
  For non-periodic: pads with fill_value at boundary
  """
  # Shift forward, but use padding instead of wrap
  f_shifted = jnp.roll(f, -1, axis=dim)
  
  # Mask the wrapped boundary value
  if dim == 0:
    f_shifted = f_shifted.at[-1, :, :].set(fill_value)
  elif dim == 1:
    f_shifted = f_shifted.at[:, -1, :].set(fill_value)
  else:  # dim == 2
    f_shifted = f_shifted.at[:, :, -1].set(fill_value)
  
  return f + f_shifted
```

However, this approach is problematic because:
1. JAX compilation would need to handle branching on BC type
2. Performance impact from conditional operations

**Better approach: Use boundary masking and correction**
- Keep periodic kernel operations
- Apply boundary corrections after each operation
- Use relaxation zones to blend solutions

### Phase 3: Integration with Time Stepping

**Modify `navier_stokes_step.py` to incorporate lateral BCs:**

```python
def step_fn(
    states: dict[str, Array],
    boundary_data: BoundaryData | None,
    cfg: config.Config,
    sg_map: dict[str, Array],
    poisson_solver: PoissonSolver,
) -> dict[str, Array]:
  """Single RK3 time step with lateral boundary forcing."""
  
  # ... existing code for RK3 integration ...
  
  # After each RK stage, apply lateral boundary conditions
  if boundary_data is not None and cfg.lateral_bcs is not None:
    states = lateral_bcs.update_lateral_boundaries(
        states,
        boundary_data,
        cfg.lateral_bcs,
        dt_float,
    )
  
  return states
```

### Phase 4: WRF Integration Workflow

**Complete workflow for WRF → Swirl-Jatmos:**

```python
# Example usage script

import jax
from swirl_jatmos import config, driver
from swirl_jatmos.boundary_conditions import (
    boundary_conditions, 
    mesoscale_forcing,
    lateral_bcs
)

# 1. Set up configuration with lateral BCs
lateral_bc_config = {
    'x_west': boundary_conditions.LateralBC(
        bc_type='relaxation',
        forcing_data_path='/path/to/wrfout_d03_2024-01-01_00:00:00',
        relaxation_width=10,
        relaxation_timescale=300.0,
    ),
    'x_east': boundary_conditions.LateralBC(bc_type='relaxation', ...),
    'y_south': boundary_conditions.LateralBC(bc_type='relaxation', ...),
    'y_north': boundary_conditions.LateralBC(bc_type='relaxation', ...),
}

cfg_ext = config.ConfigExternal(
    cx=1, cy=1, cz=1,
    nx=256, ny=256, nz=128,
    domain_x=(0, 10e3),  # 10 km LES domain
    domain_y=(0, 10e3),
    domain_z=(0, 5e3),
    dt=1.0,
    # ... other config ...
    lateral_bcs=lateral_bc_config,
)

# 2. Initialize mesoscale forcing reader
forcing_reader = mesoscale_forcing.MesoscaleForcingReader(
    wrf_file_path=lateral_bc_config['x_west'].forcing_data_path,
    les_config=config.config_from_config_external(cfg_ext),
    les_domain_center_lat=40.0,  # Example
    les_domain_center_lon=-105.0,
)

# 3. Time stepping loop with boundary updates
def run_with_mesoscale_forcing():
  states = driver.get_init_state(my_init_fn, cfg)
  
  time = 0.0
  wrf_time_idx = 0
  
  while time < t_final:
    # Update boundary data periodically (e.g., every WRF output time)
    if time % wrf_output_interval == 0:
      boundary_data = forcing_reader.read_boundary_at_time(wrf_time_idx)
      wrf_time_idx += 1
    
    # Integrate one step
    states = step_fn(states, boundary_data, cfg, ...)
    
    time += dt
```

---

## Technical Challenges & Solutions

### Challenge 1: Grid Mismatch

**Problem:** WRF and swirl-jatmos have different:
- Grid spacings (WRF typically 1-3 km, LES 10-100 m)
- Vertical coordinates (WRF: terrain-following; LES: Cartesian)
- Staggering conventions (similar but not identical)

**Solution:**
- **Vertical:** Interpolate WRF data from η (terrain-following) to LES z-levels
  - Use WRF's PH, PHB (geopotential) to get physical heights
  - Apply vertical interpolation (linear or cubic)
- **Horizontal:** Use bilinear/bicubic interpolation along boundary planes
- **Staggering:** Carefully map WRF's staggered variables to LES staggered grid
  - May need averaging (e.g., if WRF U is at different x-faces than LES u)

### Challenge 2: Variable Transformations

**Problem:** WRF and swirl-jatmos use different thermodynamic variables

**WRF variables → Swirl-Jatmos conversions:**
- `θ = T + 300`: WRF perturbation potential temp → total potential temp
- `theta_li ≈ θ - L_v/c_p * q_c`: Approximate liquid-ice potential temp
  - WRF provides QCLOUD, QICE - use these to compute theta_li
- Pressure/density: WRF provides perturbation + base state, need to combine

**Solution:** Create transformation module:

```python
def wrf_to_jatmos_thermodynamics(
    T_wrf: Array,      # Perturbation potential temp
    T_base: float,     # Base state (usually 300 K)
    qv: Array,         # Water vapor mixing ratio
    qc: Array,         # Cloud water
    qi: Array,         # Cloud ice
    wp: water.WaterParams,
) -> tuple[Array, Array]:
  """Convert WRF thermodynamic variables to swirl-jatmos.
  
  Returns:
    theta_li, q_t
  """
  theta = T_wrf + T_base
  q_t = qv + qc + qi  # Total water (assume negligible qr, qs at boundaries)
  
  # theta_li = theta - L_v/c_p * q_c - L_s/c_p * q_i
  # where L_v, L_s are latent heats
  theta_li = theta - (wp.l_v / water.CP_D) * qc - (wp.l_s / water.CP_D) * qi
  
  return theta_li, q_t
```

### Challenge 3: Map Projections & Domain Orientation

**Problem:** WRF uses map projections (Lambert Conformal, Mercator, etc.), LES uses Cartesian

**Solution:**
- Extract WRF map factors: `MAPFAC_M`, `MAPFAC_U`, `MAPFAC_V`
- If LES domain is small (<~10-20 km), can approximate as Cartesian
- For larger domains or high latitudes, need to:
  1. Rotate winds from WRF grid-relative to LES grid-relative
  2. Apply map factors to convert between physical and grid distances
  
```python
def rotate_winds_to_les_grid(
    u_wrf: Array,
    v_wrf: Array,
    wrf_grid_angle: float,  # from WRF COSALPHA, SINALPHA
) -> tuple[Array, Array]:
  """Rotate WRF winds to LES grid orientation."""
  cos_alpha = jnp.cos(wrf_grid_angle)
  sin_alpha = jnp.sin(wrf_grid_angle)
  
  u_les = u_wrf * cos_alpha + v_wrf * sin_alpha
  v_les = -u_wrf * sin_alpha + v_wrf * cos_alpha
  
  return u_les, v_les
```

### Challenge 4: Temporal Interpolation

**Problem:** WRF outputs at discrete times (e.g., every 10-60 minutes), LES timesteps are ~1-10 seconds

**Solution:**
- Cache two consecutive WRF output times
- Linear interpolation between them:

```python
class TemporalInterpolator:
  def __init__(self, forcing_reader):
    self.reader = forcing_reader
    self.bc_data_t0 = None
    self.bc_data_t1 = None
    self.time_t0 = 0.0
    self.time_t1 = 0.0
    self.current_wrf_idx = 0
    
  def get_boundary_at_time(self, t: float) -> BoundaryData:
    """Get boundary data interpolated to time t."""
    # Load next WRF time if needed
    if t > self.time_t1:
      self.bc_data_t0 = self.bc_data_t1
      self.bc_data_t1 = self.reader.read_boundary_at_time(self.current_wrf_idx + 1)
      self.time_t0 = self.time_t1
      self.time_t1 = self.bc_data_t1.time
      self.current_wrf_idx += 1
    
    # Linear interpolation
    alpha = (t - self.time_t0) / (self.time_t1 - self.time_t0)
    
    return BoundaryData(
        west_u=self.bc_data_t0.west_u * (1-alpha) + self.bc_data_t1.west_u * alpha,
        # ... interpolate all fields ...
        time=t
    )
```

### Challenge 5: Relaxation Zone Tuning

**Problem:** Need to balance:
- Too narrow relaxation zone → reflection of waves at boundary
- Too wide relaxation zone → over-constraining interior solution
- Too fast relaxation → numerical instability
- Too slow relaxation → interior drift from mesoscale forcing

**Recommendations:**
- **Width:** 10-20 grid points (1-2 km for 100m grid spacing)
- **Timescale:** 
  - Fast for mean flow: τ = 5-10 minutes
  - Slower for perturbations to allow LES eddies
  - Can use different τ for different variables
- **Weight function:** 
  - Linear ramp works well
  - Can try exponential: `w(x) = exp(-x²/σ²)`
  
**Testing:** Monitor:
- Boundary layer development in interior
- Wave reflection coefficients
- Conservation of mass (global integral of w)

---

## Alternative Approaches

### 1. **Prescribed Tendency Forcing**

Instead of relaxation zones, add forcing terms to governing equations:

```python
# In navier_stokes_step.py
def u_tendency_forcing(u: Array, u_wrf: Array, cfg: config.Config) -> Array:
  """Add WRF tendency to u-momentum equation."""
  # Only in boundary region
  mask = create_boundary_mask(cfg.lateral_bc_width)
  forcing = mask * (u_wrf - u) / cfg.forcing_timescale
  return forcing
```

**Pros:** Cleaner separation of physics and forcing  
**Cons:** Similar to relaxation, need to tune forcing strength

### 2. **Spectral Nudging**

Apply forcing only to large scales (low wavenumbers):

```python
def spectral_nudging(field: Array, field_wrf: Array, cutoff_wavelength: float):
  """Nudge only wavelengths larger than cutoff to WRF solution."""
  field_fft = jnp.fft.fft2(field, axes=(0, 1))
  wrf_fft = jnp.fft.fft2(field_wrf, axes=(0, 1))
  
  # Create low-pass filter
  filter = create_lowpass_filter(field.shape, cutoff_wavelength)
  
  # Nudge low frequencies
  nudged_fft = field_fft * (1 - filter) + wrf_fft * filter
  
  return jnp.fft.ifft2(nudged_fft, axes=(0, 1)).real
```

**Pros:** Allows LES to develop small-scale turbulence while constraining large scales  
**Cons:** More complex, requires FFTs, harder to tune

### 3. **Nested Grid with Feedback**

Two-way coupling where LES feeds back to mesoscale:

**Pros:** Most physically consistent  
**Cons:** Very complex, requires modifying both models, computational cost

---

## Implementation Checklist

- [ ] **Phase 1: Infrastructure**
  - [ ] Create `LateralBC` dataclass in `boundary_conditions.py`
  - [ ] Create `BoundaryData` dataclass for holding forcing data
  - [ ] Create `MesoscaleForcingReader` class with WRF I/O
  - [ ] Implement vertical interpolation (WRF η → LES z)
  - [ ] Implement horizontal interpolation along boundaries
  - [ ] Implement variable transformations (WRF → Jatmos)
  - [ ] Create tests for data loading and interpolation

- [ ] **Phase 2: Boundary Application**
  - [ ] Implement `apply_relaxation_zone()` function
  - [ ] Implement `apply_specified_bc()` function
  - [ ] Implement `update_lateral_boundaries()` for all variables
  - [ ] Handle staggered grid properly (u, v, w at different locations)
  - [ ] Create tests for boundary application

- [ ] **Phase 3: Integration**
  - [ ] Add `lateral_bcs` to `ConfigExternal`
  - [ ] Modify `navier_stokes_step.py` to call boundary updates
  - [ ] Modify `driver.py` to manage boundary data through time
  - [ ] Implement temporal interpolation between WRF output times
  - [ ] Handle checkpointing/restart with boundary data

- [ ] **Phase 4: Validation**
  - [ ] Test with idealized profile (uniform flow)
  - [ ] Test with WRF output from simple case (e.g., boundary layer)
  - [ ] Verify mass conservation
  - [ ] Check for spurious reflection at boundaries
  - [ ] Compare with periodic case (should match in domain interior)
  - [ ] Test different relaxation widths and timescales

- [ ] **Phase 5: Documentation & Examples**
  - [ ] Document new configuration options
  - [ ] Create example script for WRF → LES coupling
  - [ ] Add tutorial notebook
  - [ ] Document WRF variable requirements
  - [ ] Document map projection handling

---

## Example: Minimal Working Implementation

Here's a simplified end-to-end example for testing:

```python
# minimal_lateral_bc_demo.py

import jax.numpy as jnp
from swirl_jatmos import config

# 1. Simple prescribed boundary data (for testing without WRF)
class SimpleBoundaryData:
  """Prescribed boundary conditions for testing."""
  
  def __init__(self, u_west: float = 10.0):
    """Create simple uniform westerly wind."""
    self.u_west_value = u_west
  
  def get_west_u(self, ny: int, nz: int) -> jnp.ndarray:
    """Return uniform u at west boundary."""
    return jnp.ones((1, ny, nz)) * self.u_west_value


# 2. Simple relaxation application
def apply_simple_relaxation(
    field: jnp.ndarray,
    bc_value: float,
    width: int,
    tau: float,
    dt: float
) -> jnp.ndarray:
  """Apply relaxation zone at west boundary (x=0)."""
  updated = field.copy()
  
  for i in range(width):
    weight = 1.0 - i / width  # Linear ramp
    relaxation = -weight / tau * (field[i, :, :] - bc_value)
    updated = updated.at[i, :, :].add(relaxation * dt)
  
  return updated


# 3. Test in supercell case
from swirl_jatmos.sim_setups import supercell

def supercell_with_lateral_forcing():
  # Standard supercell config
  cfg_ext = config.ConfigExternal(...)  # From supercell demo
  cfg = config.config_from_config_external(cfg_ext)
  
  # Initialize
  states = supercell.init_fn(cfg)
  
  # Simple boundary data
  bc_data = SimpleBoundaryData(u_west=15.0)  # 15 m/s westerly
  
  # Time loop
  for step in range(n_steps):
    # ... RK3 integration (existing code) ...
    
    # Apply lateral BC after each step
    states['u'] = apply_simple_relaxation(
        states['u'],
        bc_data.u_west_value,
        width=10,
        tau=300.0,
        dt=cfg.dt
    )
  
  return states
```

---

## Performance Considerations

### Computational Cost

**Estimated overhead from lateral BCs:**
- Boundary data I/O: Negligible (done infrequently)
- Interpolation: ~1-2% (only on boundaries)
- Relaxation zones: ~5-10% (depends on width)

**Optimization strategies:**
1. **JIT compile** boundary application functions
2. **Batch** boundary updates (update all variables at once)
3. **Cache** interpolation weights
4. **Reduce** update frequency (update BC every N steps, not every step)

### Memory Requirements

**Additional memory:**
- Two WRF time slices: ~100 MB (for typical nested WRF domain)
- Boundary data arrays: ~10 MB (just boundary planes)
- Interpolation weights: ~1 MB

**Total:** ~100-200 MB additional (negligible compared to LES state)

---

## References & Resources

### WRF Documentation
- WRF User's Guide: https://www2.mmm.ucar.edu/wrf/users/docs/user_guide_v4/contents.html
- WRF I/O API: https://www2.mmm.ucar.edu/wrf/users/docs/api.html
- WRF ARW Tech Note: https://www2.mmm.ucar.edu/wrf/users/docs/technote/contents.html

### Relevant Literature
1. **Mirocha et al. (2014):** "Implementation of a Nonlinear Subfilter Turbulence Stress Model for Large-Eddy Simulation in the Advanced Research WRF Model" - Monthly Weather Review
2. **Muñoz-Esparza et al. (2014):** "A Fast GUI-Based Virtual Microscope for Automated Nesting of Mesoscale-to-Microscale Simulations" - Boundary-Layer Meteorology
3. **Muñoz-Esparza & Kosović (2018):** "Generation of Inflow Turbulence in Large-Eddy Simulations of Nonneutral Atmospheric Boundary Layers with the Cell Perturbation Method" - Monthly Weather Review

### Similar Implementations
- **WRF-LES Nesting:** WRF's built-in LES capability with nesting
- **PALM Model:** LES model with mesoscale forcing (COSMO, WRF)
- **MicroHH:** LES code with large-scale forcing options

---

## Next Steps

1. **Start simple:** Implement relaxation zones with prescribed (not WRF) boundary data
2. **Validate:** Test with idealized cases (uniform flow, boundary layer profiles)
3. **Add WRF I/O:** Implement `MesoscaleForcingReader` with actual WRF data
4. **Test realistic case:** Run swirl-jatmos LES nested in WRF simulation
5. **Optimize:** Profile and optimize boundary operations
6. **Document:** Create tutorials and examples

**Contact:** For WRF-specific questions, consult WRF-Users mailing list or WRF documentation.

---

## Appendix A: WRF Variable Reference

| WRF Variable | Description | Dimensions | Staggering | Units |
|--------------|-------------|------------|------------|-------|
| `U` | x-wind component | (Time, z, y, x+1) | x-staggered | m/s |
| `V` | y-wind component | (Time, z, y+1, x) | y-staggered | m/s |
| `W` | z-wind component | (Time, z+1, y, x) | z-staggered | m/s |
| `T` | Perturbation potential temp | (Time, z, y, x) | none | K |
| `P` | Perturbation pressure | (Time, z, y, x) | none | Pa |
| `PH` | Perturbation geopotential | (Time, z+1, y, x) | z-staggered | m²/s² |
| `PHB` | Base-state geopotential | (Time, z+1, y, x) | z-staggered | m²/s² |
| `QVAPOR` | Water vapor mixing ratio | (Time, z, y, x) | none | kg/kg |
| `QCLOUD` | Cloud water mixing ratio | (Time, z, y, x) | none | kg/kg |
| `QRAIN` | Rain water mixing ratio | (Time, z, y, x) | none | kg/kg |
| `QICE` | Ice mixing ratio | (Time, z, y, x) | none | kg/kg |
| `QSNOW` | Snow mixing ratio | (Time, z, y, x) | none | kg/kg |
| `T00` | Base-state temperature | scalar | none | K |
| `P00` | Base-state pressure | scalar | none | Pa |
| `XLAT` | Latitude | (Time, y, x) | none | degrees |
| `XLONG` | Longitude | (Time, y, x) | none | degrees |
| `MAPFAC_M` | Map factor (mass points) | (Time, y, x) | none | - |
| `COSALPHA` | Cosine of rotation angle | (Time, y, x) | none | - |
| `SINALPHA` | Sine of rotation angle | (Time, y, x) | none | - |

---

## Appendix B: Configuration Example

Complete configuration for LES nested in WRF:

```python
cfg_ext = config.ConfigExternal(
    # Domain decomposition
    cx=2, cy=2, cz=1,  # 4 cores total
    
    # Grid resolution
    nx=128, ny=128, nz=64,  # Per core
    # Total domain: 256 x 256 x 64 = 4.2M grid points
    
    # Physical domain (10 km x 10 km x 3 km)
    domain_x=(0, 10e3),
    domain_y=(0, 10e3),
    domain_z=(0, 3e3),
    # Grid spacing: ~40 m horizontal, ~47 m vertical
    
    # Time stepping
    dt=2.0,  # Initial timestep: 2 seconds
    timestep_control_cfg=timestep_control_config.TimestepControlConfig(
        desired_cfl=0.8,
        max_dt=5.0,
        min_dt=0.5,
    ),
    
    # Lateral boundary conditions (NEW!)
    lateral_bcs={
        'x_west': boundary_conditions.LateralBC(
            bc_type='relaxation',
            forcing_data_path='/data/wrf/wrfout_d03_2024-06-15_12:00:00',
            relaxation_width=15,  # 600 m zone
            relaxation_timescale=300.0,  # 5 minutes
            update_interval=10,  # Update every 10 timesteps
        ),
        'x_east': boundary_conditions.LateralBC(
            bc_type='relaxation',
            forcing_data_path='/data/wrf/wrfout_d03_2024-06-15_12:00:00',
            relaxation_width=15,
            relaxation_timescale=300.0,
        ),
        'y_south': boundary_conditions.LateralBC(bc_type='relaxation', ...),
        'y_north': boundary_conditions.LateralBC(bc_type='relaxation', ...),
    },
    
    # WRF domain location (NEW!)
    wrf_domain_config={
        'center_lat': 40.0,  # Latitude of LES domain center
        'center_lon': -105.0,  # Longitude
        'map_projection': 'lambert',  # WRF projection
        'wrf_output_interval': 600.0,  # WRF outputs every 10 min
    },
    
    # Vertical boundary conditions
    z_bcs=boundary_conditions.ZBoundaryConditions(
        bottom=boundary_conditions.ZBC(
            bc_type='monin_obukhov',
            mop=monin_obukhov.MoninObukhovParameters(
                surface_temperature=300.0,
                surface_moisture=0.01,
                roughness_length=0.1,
            ),
        ),
        top=boundary_conditions.ZBC(bc_type='no_flux'),
    ),
    
    # Physics
    convection_cfg=convection_config.ConvectionConfig(
        momentum_scheme='weno5_z',
        theta_li_scheme='weno5_z',
        q_t_scheme='weno5_z',
    ),
    microphysics_cfg=microphysics_config.MicrophysicsConfig(
        enable_microphysics=True,
        scheme='one_moment',
    ),
    use_sgs=True,
    viscosity=1e-3,
    diffusivity=1e-3,
    
    # Output
    aux_output_fields=('q_c', 'q_i', 'w_max'),
    checkpoint_cycle_interval=1,
)
```

This configuration creates an LES domain forced by WRF on all four lateral boundaries, suitable for simulating convective storms, boundary layer turbulence, or other mesoscale-to-microscale phenomena.

