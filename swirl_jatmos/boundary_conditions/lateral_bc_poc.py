# Copyright 2024 The swirl_jatmos Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Proof-of-concept for lateral boundary conditions from mesoscale models.

This module demonstrates the core functionality needed to apply lateral
boundary conditions from mesoscale weather models (e.g., WRF) to swirl-jatmos.

This is a simplified proof-of-concept. A production implementation would need:
1. Proper WRF I/O with netCDF4
2. Vertical coordinate transformations (WRF eta -> LES z)
3. Horizontal interpolation (WRF grid spacing -> LES grid spacing)
4. Map projection handling
5. Variable transformations (WRF thermodynamics -> JAX thermodynamics)
"""

import dataclasses
from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp

Array: TypeAlias = jax.Array


@dataclasses.dataclass(frozen=True)
class RelaxationZoneConfig:
  """Configuration for relaxation zone (nudging) boundary conditions."""
  
  # Width of the relaxation zone in grid points
  width: int = 10
  
  # Relaxation timescale in seconds (τ in: dφ/dt = -(φ - φ_BC)/τ)
  timescale: float = 300.0
  
  # Weight function: 'linear', 'exponential', 'cosine'
  weight_function: Literal['linear', 'exponential', 'cosine'] = 'linear'


def create_relaxation_weights(
    width: int,
    weight_function: str = 'linear',
) -> Array:
  """Create weight function for relaxation zone.
  
  Weight = 1 at boundary (full nudging), 0 at edge of zone (no nudging).
  
  Args:
    width: Number of grid points in relaxation zone.
    weight_function: Type of weight function ('linear', 'exponential', 'cosine').
  
  Returns:
    1D array of weights from 1.0 to 0.0.
  """
  x = jnp.linspace(0, 1, width)  # Normalized distance: 0 at BC, 1 at edge
  
  if weight_function == 'linear':
    weights = 1.0 - x
  elif weight_function == 'exponential':
    # Exponential decay: w(x) = exp(-4*x) (chosen so w(1) ≈ 0.018)
    weights = jnp.exp(-4.0 * x)
  elif weight_function == 'cosine':
    # Smooth cosine taper: w(x) = 0.5 * (1 + cos(π*x))
    weights = 0.5 * (1.0 + jnp.cos(jnp.pi * x))
  else:
    raise ValueError(f'Unknown weight function: {weight_function}')
  
  return weights


def apply_relaxation_west(
    field: Array,
    bc_value: Array,
    config: RelaxationZoneConfig,
    dt: float,
) -> Array:
  """Apply relaxation zone at western (x_min) boundary.
  
  Applies Newtonian relaxation (nudging):
    φ^(n+1) = φ^n - (dt/τ) * w(x) * (φ^n - φ_BC)
  
  where w(x) is the weight function (1 at boundary, 0 at edge of zone).
  
  Args:
    field: 3D field to apply relaxation to, shape (nx, ny, nz).
    bc_value: Boundary condition values, shape (1, ny, nz) or (ny, nz).
    config: Relaxation zone configuration.
    dt: Time step in seconds.
  
  Returns:
    Updated field with relaxation applied.
  """
  # Create weights: shape (width,)
  weights = create_relaxation_weights(config.width, config.weight_function)
  
  # Reshape for broadcasting: (width, 1, 1)
  weights_3d = weights.reshape(-1, 1, 1)
  
  # Ensure bc_value is 2D: (ny, nz)
  if bc_value.ndim == 3:
    bc_value = bc_value.squeeze(axis=0)
  
  # Extract relaxation zone from field: (width, ny, nz)
  zone = field[:config.width, :, :]
  
  # Compute relaxation tendency
  tendency = -weights_3d / config.timescale * (zone - bc_value[jnp.newaxis, :, :])
  
  # Apply update: φ^(n+1) = φ^n + dt * tendency
  updated_zone = zone + dt * tendency
  
  # Update field in-place (JAX functional style)
  updated_field = field.at[:config.width, :, :].set(updated_zone)
  
  return updated_field


def apply_relaxation_east(
    field: Array,
    bc_value: Array,
    config: RelaxationZoneConfig,
    dt: float,
) -> Array:
  """Apply relaxation zone at eastern (x_max) boundary.
  
  Similar to apply_relaxation_west but for the opposite boundary.
  Weight function is reversed: 1 at x_max, 0 inward.
  """
  weights = create_relaxation_weights(config.width, config.weight_function)
  
  # Reverse weights (1 at far boundary, 0 inward)
  weights = weights[::-1]
  weights_3d = weights.reshape(-1, 1, 1)
  
  if bc_value.ndim == 3:
    bc_value = bc_value.squeeze(axis=0)
  
  # Extract eastern relaxation zone
  zone = field[-config.width:, :, :]
  
  tendency = -weights_3d / config.timescale * (zone - bc_value[jnp.newaxis, :, :])
  updated_zone = zone + dt * tendency
  
  updated_field = field.at[-config.width:, :, :].set(updated_zone)
  
  return updated_field


def apply_relaxation_south(
    field: Array,
    bc_value: Array,
    config: RelaxationZoneConfig,
    dt: float,
) -> Array:
  """Apply relaxation zone at southern (y_min) boundary."""
  weights = create_relaxation_weights(config.width, config.weight_function)
  weights_3d = weights.reshape(1, -1, 1)  # Shape: (1, width, 1)
  
  if bc_value.ndim == 3:
    bc_value = bc_value.squeeze(axis=1)
  
  zone = field[:, :config.width, :]
  
  tendency = -weights_3d / config.timescale * (zone - bc_value[:, jnp.newaxis, :])
  updated_zone = zone + dt * tendency
  
  updated_field = field.at[:, :config.width, :].set(updated_zone)
  
  return updated_field


def apply_relaxation_north(
    field: Array,
    bc_value: Array,
    config: RelaxationZoneConfig,
    dt: float,
) -> Array:
  """Apply relaxation zone at northern (y_max) boundary."""
  weights = create_relaxation_weights(config.width, config.weight_function)
  weights = weights[::-1]
  weights_3d = weights.reshape(1, -1, 1)
  
  if bc_value.ndim == 3:
    bc_value = bc_value.squeeze(axis=1)
  
  zone = field[:, -config.width:, :]
  
  tendency = -weights_3d / config.timescale * (zone - bc_value[:, jnp.newaxis, :])
  updated_zone = zone + dt * tendency
  
  updated_field = field.at[:, -config.width:, :].set(updated_zone)
  
  return updated_field


def apply_all_lateral_boundaries(
    field: Array,
    bc_west: Array | None,
    bc_east: Array | None,
    bc_south: Array | None,
    bc_north: Array | None,
    config: RelaxationZoneConfig,
    dt: float,
) -> Array:
  """Apply relaxation zones to all four lateral boundaries.
  
  Args:
    field: 3D field, shape (nx, ny, nz).
    bc_west: Western boundary values, shape (1, ny, nz) or None.
    bc_east: Eastern boundary values, shape (1, ny, nz) or None.
    bc_south: Southern boundary values, shape (nx, 1, nz) or None.
    bc_north: Northern boundary values, shape (nx, 1, nz) or None.
    config: Relaxation zone configuration.
    dt: Time step.
  
  Returns:
    Field with all lateral boundaries updated.
  """
  updated = field
  
  if bc_west is not None:
    updated = apply_relaxation_west(updated, bc_west, config, dt)
  
  if bc_east is not None:
    updated = apply_relaxation_east(updated, bc_east, config, dt)
  
  if bc_south is not None:
    updated = apply_relaxation_south(updated, bc_south, config, dt)
  
  if bc_north is not None:
    updated = apply_relaxation_north(updated, bc_north, config, dt)
  
  return updated


# ============================================================================
# Example: Prescribed boundary conditions for testing
# ============================================================================


def create_prescribed_westerly_wind(
    ny: int,
    nz: int,
    u_surface: float = 10.0,
    u_top: float = 20.0,
) -> Array:
  """Create a prescribed westerly wind profile.
  
  Linear increase from surface to top of domain.
  Useful for testing without actual WRF data.
  
  Args:
    ny: Number of grid points in y.
    nz: Number of grid points in z.
    u_surface: Wind speed at surface [m/s].
    u_top: Wind speed at top [m/s].
  
  Returns:
    Wind profile, shape (ny, nz).
  """
  # Create vertical profile (increase with height)
  z_normalized = jnp.linspace(0, 1, nz)
  u_profile = u_surface + (u_top - u_surface) * z_normalized
  
  # Broadcast to (ny, nz)
  u_boundary = jnp.tile(u_profile, (ny, 1))
  
  return u_boundary


def create_prescribed_boundary_layer_profile(
    ny: int,
    nz: int,
    z_coords: Array,
    theta_surface: float = 300.0,
    theta_top: float = 310.0,
    q_surface: float = 0.012,
    q_top: float = 0.001,
) -> tuple[Array, Array]:
  """Create prescribed profiles for potential temperature and humidity.
  
  Args:
    ny: Number of grid points in y.
    nz: Number of grid points in z.
    z_coords: Vertical coordinates [m].
    theta_surface: Surface potential temperature [K].
    theta_top: Top potential temperature [K].
    q_surface: Surface specific humidity [kg/kg].
    q_top: Top specific humidity [kg/kg].
  
  Returns:
    Tuple of (theta_boundary, q_boundary), each shape (ny, nz).
  """
  # Assume z_coords is 1D array of length nz
  z_norm = (z_coords - z_coords[0]) / (z_coords[-1] - z_coords[0])
  
  # Potential temperature: linear stratification
  theta_profile = theta_surface + (theta_top - theta_surface) * z_norm
  
  # Humidity: exponential decay
  # q(z) = q_top + (q_surface - q_top) * exp(-z/z_scale)
  z_scale = 1500.0  # [m]
  q_profile = q_top + (q_surface - q_top) * jnp.exp(-z_coords / z_scale)
  
  # Broadcast to (ny, nz)
  theta_boundary = jnp.tile(theta_profile, (ny, 1))
  q_boundary = jnp.tile(q_profile, (ny, 1))
  
  return theta_boundary, q_boundary


# ============================================================================
# Demonstration / Test Function
# ============================================================================


def demo_relaxation_zone():
  """Demonstrate relaxation zone application."""
  
  # Create a simple 3D field (e.g., u-velocity)
  nx, ny, nz = 100, 80, 40
  
  # Initialize with some interior flow (e.g., random turbulence)
  key = jax.random.PRNGKey(42)
  u_field = 5.0 + jax.random.normal(key, (nx, ny, nz)) * 2.0
  
  # Create prescribed boundary condition (westerly wind)
  u_bc_west = create_prescribed_westerly_wind(ny, nz, u_surface=10.0, u_top=20.0)
  
  # Relaxation zone configuration
  config = RelaxationZoneConfig(
      width=10,
      timescale=300.0,  # 5 minutes
      weight_function='cosine',
  )
  
  # Simulate applying relaxation over multiple timesteps
  dt = 2.0  # 2 seconds
  n_steps = 100
  
  print('Demonstrating lateral boundary condition relaxation...')
  print(f'Domain size: {nx} x {ny} x {nz}')
  print(f'Relaxation zone width: {config.width} grid points')
  print(f'Relaxation timescale: {config.timescale} s')
  print(f'Time step: {dt} s')
  print(f'Number of steps: {n_steps}')
  print()
  
  u_updated = u_field
  for step in range(n_steps):
    u_updated = apply_relaxation_west(
        u_updated,
        u_bc_west,
        config,
        dt,
    )
    
    # Print diagnostics every 20 steps
    if step % 20 == 0:
      u_interior_mean = jnp.mean(u_updated[20:, :, :])
      u_boundary_mean = jnp.mean(u_updated[:5, :, :])
      print(f'Step {step:3d}: '
            f'Boundary mean = {u_boundary_mean:.2f} m/s, '
            f'Interior mean = {u_interior_mean:.2f} m/s')
  
  print()
  print('Final state:')
  print(f'  u at west boundary (x=0): {jnp.mean(u_updated[0, :, :]):.2f} m/s')
  print(f'  u at edge of zone (x={config.width}): '
        f'{jnp.mean(u_updated[config.width, :, :]):.2f} m/s')
  print(f'  u in interior (x={nx//2}): '
        f'{jnp.mean(u_updated[nx//2, :, :]):.2f} m/s')
  print()
  
  # Check that boundary has relaxed toward prescribed value
  target_mean = jnp.mean(u_bc_west)
  actual_mean = jnp.mean(u_updated[0, :, :])
  print(f'Prescribed BC mean: {target_mean:.2f} m/s')
  print(f'Actual boundary mean: {actual_mean:.2f} m/s')
  print(f'Difference: {abs(target_mean - actual_mean):.2f} m/s')
  
  return u_updated


if __name__ == '__main__':
  # Run demonstration
  demo_relaxation_zone()
  
  print('\n' + '='*70)
  print('Proof-of-concept demonstration complete!')
  print('='*70)
  print()
  print('Next steps for production implementation:')
  print('1. Create WRF data reader (netCDF4 I/O)')
  print('2. Implement vertical coordinate transformation (WRF eta -> LES z)')
  print('3. Add horizontal interpolation (WRF grid -> LES grid)')
  print('4. Handle map projections and rotations')
  print('5. Transform WRF variables to swirl-jatmos variables')
  print('6. Integrate with swirl-jatmos time stepping loop')
  print('7. Add temporal interpolation between WRF output times')
  print('8. Validate with test cases')

