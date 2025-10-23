#!/usr/bin/env python3
"""Example: Running swirl-jatmos LES nested within WRF mesoscale simulation.

This example demonstrates how to set up and run a swirl-jatmos large-eddy
simulation (LES) with lateral boundary conditions provided by WRF mesoscale
model output.

Scenario:
---------
- WRF runs at 3 km grid spacing covering a large domain
- Swirl-jatmos runs at 50 m grid spacing covering a 10 km x 10 km subdomain
- WRF provides lateral boundary conditions every 10 minutes
- LES uses relaxation zones to blend WRF forcing with interior solution

Requirements:
-------------
- WRF output file (wrfout_d0X_YYYY-MM-DD_HH:MM:SS)
- Swirl-jatmos with lateral BC capability (see MESOSCALE_BC_INTEGRATION_GUIDE.md)

Note: This is a conceptual example. Full implementation requires the lateral
BC infrastructure described in the integration guide.
"""

import sys
from pathlib import Path

from absl import app, flags
import jax
import jax.numpy as jnp
import numpy as np

# Swirl-jatmos imports (adjust as needed when lateral BC is implemented)
from swirl_jatmos import config
from swirl_jatmos import convection_config
from swirl_jatmos import driver
from swirl_jatmos import sim_initializer
from swirl_jatmos import timestep_control_config
from swirl_jatmos.boundary_conditions import boundary_conditions
from swirl_jatmos.boundary_conditions import monin_obukhov
from swirl_jatmos.thermodynamics import water

# These would be new modules (not yet implemented)
# from swirl_jatmos.boundary_conditions import lateral_bcs
# from swirl_jatmos.boundary_conditions import mesoscale_forcing

# Enable JAX 64-bit mode (required for swirl-jatmos)
jax.config.update('jax_enable_x64', True)

FLAGS = flags.FLAGS

# Command-line flags
flags.DEFINE_string(
    'wrf_file',
    '/path/to/wrfout_d03_2024-06-15_12:00:00',
    'Path to WRF output file for boundary forcing'
)
flags.DEFINE_float('les_center_lat', 40.0, 'Latitude of LES domain center')
flags.DEFINE_float('les_center_lon', -105.0, 'Longitude of LES domain center')
flags.DEFINE_float('t_final', 3600.0, 'Simulation duration [seconds]')
flags.DEFINE_string('output_dir', '/tmp/wrf_nested_les', 'Output directory')


def create_les_config_from_wrf() -> config.ConfigExternal:
  """Create swirl-jatmos configuration for WRF-nested LES.
  
  This configuration is designed to match a typical WRF nested domain:
  - WRF domain 3 (d03): 3 km grid spacing
  - LES domain: 50 m grid spacing
  - LES covers 10 km x 10 km subdomain within WRF d03
  """
  
  # Domain size
  # 10 km x 10 km horizontal, 3 km vertical
  lx = 10e3  # [m]
  ly = 10e3  # [m]
  lz = 3e3   # [m]
  
  # Grid resolution
  # 50 m horizontal resolution: 10000m / 50m = 200 points
  # 50 m vertical resolution: 3000m / 50m = 60 points
  nx_total = 200
  ny_total = 200
  nz_total = 60
  
  # Domain decomposition (adjust based on available TPU/GPU cores)
  # For 8 cores: 2x2x2 = 8
  cx, cy, cz = 2, 2, 2
  
  # Points per core
  nx = nx_total // cx  # 100
  ny = ny_total // cy  # 100
  nz = nz_total // cz  # 30
  
  # Time stepping
  # CFL condition: dt < dx / (u_max)
  # For 50 m grid and ~30 m/s winds: dt < 1.6 s
  dt_initial = 1.0  # Start conservative
  
  # Lateral boundary conditions (NEW - conceptual, needs implementation)
  # In production, this would be:
  # lateral_bcs = {
  #     'x_west': boundary_conditions.LateralBC(
  #         bc_type='relaxation',
  #         forcing_data_path=FLAGS.wrf_file,
  #         relaxation_width=15,  # 750 m zone
  #         relaxation_timescale=300.0,  # 5 minutes
  #         update_interval=10,
  #     ),
  #     ... similar for east, south, north
  # }
  
  cfg_ext = config.ConfigExternal(
      # Parallelization
      cx=cx,
      cy=cy,
      cz=cz,
      
      # Grid size (per core)
      nx=nx,
      ny=ny,
      nz=nz,
      
      # Physical domain
      domain_x=(0.0, lx),
      domain_y=(0.0, ly),
      domain_z=(0.0, lz),
      
      # Time control
      dt=dt_initial,
      timestep_control_cfg=timestep_control_config.TimestepControlConfig(
          desired_cfl=0.8,
          max_dt=2.0,
          min_dt=0.2,
          max_change_factor=1.2,
          update_interval_steps=5,
      ),
      
      # Vertical boundary conditions
      z_bcs=boundary_conditions.ZBoundaryConditions(
          bottom=boundary_conditions.ZBC(
              bc_type='monin_obukhov',
              mop=monin_obukhov.MoninObukhovParameters(
                  surface_temperature=300.0,  # Would be from WRF
                  surface_moisture=0.01,       # Would be from WRF
                  roughness_length=0.1,        # 10 cm (short grass/crops)
                  use_fixed_exchange_coefficients=False,
              ),
          ),
          top=boundary_conditions.ZBC(
              bc_type='no_flux',
          ),
      ),
      
      # Convection scheme
      convection_cfg=convection_config.ConvectionConfig(
          momentum_scheme='weno5_z',
          theta_li_scheme='weno5_z',
          q_t_scheme='weno5_z',
      ),
      
      # Microphysics (single-moment)
      microphysics_cfg=config.microphysics_config.MicrophysicsConfig(
          enable_microphysics=True,
          enable_warm_rain=True,
          enable_ice=True,
      ),
      
      # Turbulence
      use_sgs=True,
      viscosity=1e-3,      # Will be augmented by SGS model
      diffusivity=1e-3,
      enforce_max_diffusivity=True,
      
      # Output
      aux_output_fields=('q_c', 'q_i', 'w_max'),
      diagnostic_fields=('tke', 'buoyancy_flux'),
      checkpoint_cycle_interval=1,
      disable_checkpointing=False,
  )
  
  return cfg_ext


def initialize_from_wrf(cfg: config.Config) -> dict[str, jax.Array]:
  """Initialize LES state from WRF data.
  
  In a full implementation, this would:
  1. Read WRF initial time slice
  2. Interpolate WRF data to LES grid (vertical and horizontal)
  3. Transform WRF variables to swirl-jatmos variables
  4. Add perturbations for LES turbulence spin-up
  
  For now, this is a placeholder that creates a simple initial state.
  """
  
  # Get grid
  grid_map = sim_initializer.initialize_grids(cfg)
  x_c = grid_map['x_c']
  y_c = grid_map['y_c']
  z_c = grid_map['z_c']
  
  nx, ny, nz = len(cfg.x_c), len(cfg.y_c), len(cfg.z_c)
  
  # In production, would read from WRF:
  # wrf_data = mesoscale_forcing.read_wrf_initial_condition(FLAGS.wrf_file)
  # u_init, v_init, w_init, theta_init, q_init = wrf_data.interpolate_to_les_grid(cfg)
  
  # For now, create simple profiles
  # Westerly wind increasing with height
  u_profile = 10.0 + 10.0 * (z_c / cfg.domain_z[1])
  u_init = jnp.tile(u_profile, (nx, ny, 1))
  
  # Small meridional component
  v_profile = 2.0 * jnp.ones_like(z_c)
  v_init = jnp.tile(v_profile, (nx, ny, 1))
  
  # Initially no vertical motion
  w_init = jnp.zeros((nx, ny, nz))
  
  # Potential temperature (stable stratification)
  theta_profile = 300.0 + 5.0 * (z_c / cfg.domain_z[1])
  theta_init = jnp.tile(theta_profile, (nx, ny, 1))
  
  # Humidity (decreasing with height)
  q_profile = 0.012 * jnp.exp(-z_c / 1500.0)
  q_init = jnp.tile(q_profile, (nx, ny, 1))
  
  # Add small random perturbations to trigger turbulence
  key = jax.random.PRNGKey(42)
  key_u, key_v, key_theta = jax.random.split(key, 3)
  
  u_init += jax.random.normal(key_u, u_init.shape) * 0.5
  v_init += jax.random.normal(key_v, v_init.shape) * 0.5
  theta_init += jax.random.normal(key_theta, theta_init.shape) * 0.1
  
  # Reference state (hydrostatic balance)
  # This is simplified - should match WRF's reference state
  p_ref = 100000.0 * (1.0 - 0.0065 * z_c / 288.15) ** 5.256
  rho_ref = p_ref / (water.R_D * theta_init[0, 0, :])
  
  rho_ref_3d = jnp.tile(rho_ref, (nx, ny, 1))
  p_ref_3d = jnp.tile(p_ref, (nx, ny, 1))
  
  # Shard arrays
  shard_3d = lambda x: sim_initializer.shard_3d(x, cfg)
  
  states = {
      'u': shard_3d(u_init),
      'v': shard_3d(v_init),
      'w': shard_3d(w_init),
      'theta_li_0': sim_initializer.shard_arr(theta_init[0, 0, :], cfg, 'z'),
      'dtheta_li': shard_3d(theta_init - theta_init[0, 0, :]),
      'q_t': shard_3d(q_init),
      'q_r': shard_3d(jnp.zeros_like(u_init)),
      'q_s': shard_3d(jnp.zeros_like(u_init)),
      'p': shard_3d(jnp.zeros_like(u_init)),
      'p_ref_xxc': sim_initializer.shard_arr(p_ref, cfg, 'z'),
      'rho_xxc': shard_3d(rho_ref_3d),
      'rho_xxf': shard_3d(rho_ref_3d),  # Simplified
  }
  
  return states


def run_wrf_nested_les(argv):
  """Main function to run WRF-nested LES."""
  
  print('='*70)
  print('WRF-Nested LES Simulation with Swirl-Jatmos')
  print('='*70)
  print()
  print('Configuration:')
  print(f'  WRF forcing file: {FLAGS.wrf_file}')
  print(f'  LES domain center: ({FLAGS.les_center_lat}°N, {FLAGS.les_center_lon}°E)')
  print(f'  Simulation duration: {FLAGS.t_final} seconds ({FLAGS.t_final/3600:.1f} hours)')
  print(f'  Output directory: {FLAGS.output_dir}')
  print()
  
  # Create configuration
  print('Setting up LES configuration...')
  cfg_ext = create_les_config_from_wrf()
  cfg = config.config_from_config_external(cfg_ext)
  
  print(f'  Domain size: {cfg.domain_x[1]/1e3:.1f} x {cfg.domain_y[1]/1e3:.1f} x {cfg.domain_z[1]/1e3:.1f} km')
  print(f'  Grid points: {len(cfg.x_c)} x {len(cfg.y_c)} x {len(cfg.z_c)}')
  print(f'  Grid spacing: ~{cfg.grid_spacings[0]:.1f} x {cfg.grid_spacings[1]:.1f} x {cfg.grid_spacings[2]:.1f} m')
  print(f'  Initial timestep: {cfg.dt:.2f} s')
  print()
  
  # Save configuration
  config.save_json(cfg_ext, FLAGS.output_dir)
  
  # Initialize mesoscale forcing reader (would be implemented)
  # print('Initializing WRF boundary forcing...')
  # forcing_reader = mesoscale_forcing.MesoscaleForcingReader(
  #     wrf_file_path=FLAGS.wrf_file,
  #     les_config=cfg,
  #     les_domain_center_lat=FLAGS.les_center_lat,
  #     les_domain_center_lon=FLAGS.les_center_lon,
  # )
  print('NOTE: WRF forcing not yet implemented - using prescribed BCs')
  print()
  
  # Run simulation
  print('Starting simulation...')
  print()
  
  # In production, would use:
  # states, aux_output, diagnostics = driver.run_driver_with_lateral_forcing(
  #     initialize_from_wrf,
  #     forcing_reader,
  #     FLAGS.output_dir,
  #     FLAGS.t_final,
  #     sec_per_cycle=600.0,  # Output every 10 minutes
  #     cfg=cfg,
  # )
  
  # For now, just initialize to demonstrate
  states = initialize_from_wrf(cfg)
  
  print('Initial state created successfully!')
  print(f"  u range: [{np.min(states['u']):.2f}, {np.max(states['u']):.2f}] m/s")
  print(f"  v range: [{np.min(states['v']):.2f}, {np.max(states['v']):.2f}] m/s")
  print(f"  theta range: [{np.min(states['theta_li_0']):.2f}, {np.max(states['theta_li_0']):.2f}] K")
  print(f"  q_t range: [{np.min(states['q_t'])*1000:.2f}, {np.max(states['q_t'])*1000:.2f}] g/kg")
  print()
  
  print('='*70)
  print('NOTE: This is a demonstration of the configuration.')
  print('Full implementation requires lateral BC infrastructure.')
  print('See MESOSCALE_BC_INTEGRATION_GUIDE.md for details.')
  print('='*70)
  
  return 0


def main(argv):
  """Entry point."""
  del argv  # Unused
  return run_wrf_nested_les(argv)


if __name__ == '__main__':
  app.run(main)

