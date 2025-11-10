"""Observation operator utilities for radar reflectivity

This module provides observation operators that transform model state variables
into observable quantities (e.g., radar reflectivity).
"""

import warnings
from typing import Optional, Union, Tuple

import numpy as np
import jax.numpy as jnp
from jax import Array


class RadarReflectivityOperator:
    """Observation operator for simulating radar reflectivity from model state.
    
    This class transforms model state variables (hydrometeor mixing ratios,
    temperature, pressure) into simulated radar reflectivity values using
    standard meteorological relationships.
    
    Attributes:
        wavelength (float): Radar wavelength in meters. Common values:
            - 0.10 m (S-band, WSR-88D)
            - 0.05 m (C-band)
            - 0.03 m (X-band)
            Default is 0.10 (S-band).
        frequency (float): Radar frequency in GHz, computed from wavelength.
        rain_scheme (str): Rain reflectivity calculation scheme. Options:
            - 'marshall_palmer': Marshall-Palmer Z-R relationship
            - 'exponential': Exponential drop size distribution
            Default is 'marshall_palmer'.
        snow_scheme (str): Snow reflectivity calculation scheme. Options:
            - 'sekhon_srivastava': Standard snow relationship
            - 'gunn_marshall': Gunn-Marshall relationship
            Default is 'sekhon_srivastava'.
        graupel_scheme (str): Graupel reflectivity calculation scheme.
            Default is 'sekhon_srivastava'.
        temp_threshold_rain (float): Temperature threshold (K) above which
            precipitation is treated as rain. Default is 273.15 K.
        temp_threshold_snow (float): Temperature threshold (K) below which
            precipitation is treated as snow. Default is 271.15 K.
        include_rain (bool): Include rain contribution to reflectivity.
            Default is True.
        include_snow (bool): Include snow contribution to reflectivity.
            Default is True.
        include_graupel (bool): Include graupel contribution to reflectivity.
            Default is True.
        include_cloud (bool): Include cloud water contribution to reflectivity.
            Default is False (cloud droplets typically too small).
        min_reflectivity (float): Minimum detectable reflectivity in dBZ.
            Values below this are set to this threshold. Default is -30 dBZ.
        max_reflectivity (float): Maximum reflectivity in dBZ for saturation.
            Default is 80 dBZ.
        use_jax (bool): Use JAX arrays for computations instead of NumPy.
            Default is False.
        
    References:
        Marshall, J. S., & Palmer, W. M. (1948). The distribution of raindrops
            with size. Journal of Meteorology, 5(4), 165-166.
        Sekhon, R. S., & Srivastava, R. C. (1970). Snow size spectra and radar
            reflectivity. Journal of the Atmospheric Sciences, 27(2), 299-307.
    """
    
    def __init__(
        self,
        wavelength: float = 0.10,
        rain_scheme: str = 'marshall_palmer',
        snow_scheme: str = 'sekhon_srivastava',
        graupel_scheme: str = 'sekhon_srivastava',
        temp_threshold_rain: float = 273.15,
        temp_threshold_snow: float = 271.15,
        include_rain: bool = True,
        include_snow: bool = True,
        include_graupel: bool = True,
        include_cloud: bool = False,
        min_reflectivity: float = -30.0,
        max_reflectivity: float = 80.0,
        use_jax: bool = False,
    ):
        """Initialize the radar reflectivity observation operator.
        
        Args:
            wavelength: Radar wavelength in meters
            rain_scheme: Rain reflectivity calculation scheme
            snow_scheme: Snow reflectivity calculation scheme
            graupel_scheme: Graupel reflectivity calculation scheme
            temp_threshold_rain: Temperature threshold for rain (K)
            temp_threshold_snow: Temperature threshold for snow (K)
            include_rain: Include rain in reflectivity calculation
            include_snow: Include snow in reflectivity calculation
            include_graupel: Include graupel in reflectivity calculation
            include_cloud: Include cloud water in reflectivity calculation
            min_reflectivity: Minimum detectable reflectivity (dBZ)
            max_reflectivity: Maximum reflectivity threshold (dBZ)
            use_jax: Use JAX for computations
        """
        self.wavelength = wavelength
        self.frequency = 3e8 / wavelength / 1e9  # Convert to GHz
        
        self.rain_scheme = rain_scheme
        self.snow_scheme = snow_scheme
        self.graupel_scheme = graupel_scheme
        
        self.temp_threshold_rain = temp_threshold_rain
        self.temp_threshold_snow = temp_threshold_snow
        
        self.include_rain = include_rain
        self.include_snow = include_snow
        self.include_graupel = include_graupel
        self.include_cloud = include_cloud
        
        self.min_reflectivity = min_reflectivity
        self.max_reflectivity = max_reflectivity
        
        self.use_jax = use_jax
        self._array_module = jnp if use_jax else np
        
        # Constants
        self.rho_air_sealevel = 1.225  # kg/m^3
        self.eps = 1e-10  # Small value to avoid division by zero
        
    def _linear_to_dbz(self, z_linear: Union[np.ndarray, Array]) -> Union[np.ndarray, Array]:
        """Convert linear reflectivity (mm^6/m^3) to dBZ.
        
        Args:
            z_linear: Linear reflectivity values
            
        Returns:
            Reflectivity in dBZ
        """
        xp = self._array_module
        z_dbz = 10.0 * xp.log10(xp.maximum(z_linear, self.eps))
        return xp.clip(z_dbz, self.min_reflectivity, self.max_reflectivity)
    
    def _dbz_to_linear(self, z_dbz: Union[np.ndarray, Array]) -> Union[np.ndarray, Array]:
        """Convert dBZ to linear reflectivity (mm^6/m^3).
        
        Args:
            z_dbz: Reflectivity in dBZ
            
        Returns:
            Linear reflectivity values
        """
        xp = self._array_module
        return 10.0 ** (z_dbz / 10.0)
    
    def _compute_air_density(
        self,
        temperature: Union[np.ndarray, Array],
        pressure: Union[np.ndarray, Array]
    ) -> Union[np.ndarray, Array]:
        """Compute air density from temperature and pressure.
        
        Uses the ideal gas law: rho = P / (R_d * T)
        
        Args:
            temperature: Temperature in Kelvin
            pressure: Pressure in Pa
            
        Returns:
            Air density in kg/m^3
        """
        R_d = 287.05  # Specific gas constant for dry air (J/kg/K)
        return pressure / (R_d * temperature)
    
    def _reflectivity_rain_marshall_palmer(
        self,
        qr: Union[np.ndarray, Array],
        rho_air: Union[np.ndarray, Array]
    ) -> Union[np.ndarray, Array]:
        """Compute rain reflectivity using Marshall-Palmer relationship.
        
        Z = 3.63e9 * (rho_air * qr)^1.75
        
        Args:
            qr: Rain mixing ratio (kg/kg)
            rho_air: Air density (kg/m^3)
            
        Returns:
            Linear reflectivity (mm^6/m^3)
        """
        xp = self._array_module
        # Convert mixing ratio to rain water content (g/m^3)
        rainwater_content = rho_air * qr * 1000.0  # kg/m^3 to g/m^3
        
        # Marshall-Palmer Z-R relationship constants
        a = 3.63e9
        b = 1.75
        
        z_linear = a * xp.power(xp.maximum(rainwater_content, self.eps), b)
        return z_linear
    
    def _reflectivity_snow_sekhon_srivastava(
        self,
        qs: Union[np.ndarray, Array],
        rho_air: Union[np.ndarray, Array]
    ) -> Union[np.ndarray, Array]:
        """Compute snow reflectivity using Sekhon-Srivastava relationship.
        
        Args:
            qs: Snow mixing ratio (kg/kg)
            rho_air: Air density (kg/m^3)
            
        Returns:
            Linear reflectivity (mm^6/m^3)
        """
        xp = self._array_module
        # Convert mixing ratio to snow water content (g/m^3)
        snowwater_content = rho_air * qs * 1000.0
        
        # Sekhon-Srivastava relationship constants
        a = 2.53e12
        b = 2.0
        
        z_linear = a * xp.power(xp.maximum(snowwater_content, self.eps), b)
        return z_linear
    
    def _reflectivity_graupel(
        self,
        qg: Union[np.ndarray, Array],
        rho_air: Union[np.ndarray, Array]
    ) -> Union[np.ndarray, Array]:
        """Compute graupel reflectivity.
        
        Args:
            qg: Graupel mixing ratio (kg/kg)
            rho_air: Air density (kg/m^3)
            
        Returns:
            Linear reflectivity (mm^6/m^3)
        """
        xp = self._array_module
        # Convert mixing ratio to graupel water content (g/m^3)
        graupelwater_content = rho_air * qg * 1000.0
        
        # Use similar relationship to snow but with different constants
        a = 4.26e12
        b = 1.9
        
        z_linear = a * xp.power(xp.maximum(graupelwater_content, self.eps), b)
        return z_linear
    
    def _reflectivity_cloud(
        self,
        qc: Union[np.ndarray, Array],
        rho_air: Union[np.ndarray, Array]
    ) -> Union[np.ndarray, Array]:
        """Compute cloud water reflectivity.
        
        Note: Cloud droplets are typically small and contribute minimally
        to radar reflectivity at most wavelengths.
        
        Args:
            qc: Cloud water mixing ratio (kg/kg)
            rho_air: Air density (kg/m^3)
            
        Returns:
            Linear reflectivity (mm^6/m^3)
        """
        xp = self._array_module
        cloudwater_content = rho_air * qc * 1000.0
        
        # Simplified relationship for cloud droplets
        a = 1e6
        b = 2.0
        
        z_linear = a * xp.power(xp.maximum(cloudwater_content, self.eps), b)
        return z_linear
    
    def forward(
        self,
        state: dict,
        return_dbz: bool = True
    ) -> Union[np.ndarray, Array]:
        """Apply forward observation operator to model state.
        
        This is the main method that transforms model state variables into
        simulated radar reflectivity observations.
        
        Args:
            state: Dictionary containing model state variables. Expected keys:
                - 'temperature': Temperature in K (required)
                - 'pressure': Pressure in Pa (required)
                - 'qr': Rain mixing ratio in kg/kg (optional)
                - 'qs': Snow mixing ratio in kg/kg (optional)
                - 'qg': Graupel mixing ratio in kg/kg (optional)
                - 'qc': Cloud water mixing ratio in kg/kg (optional)
            return_dbz: If True, return reflectivity in dBZ. If False, return
                linear reflectivity (mm^6/m^3). Default is True.
                
        Returns:
            Simulated radar reflectivity, either in dBZ or linear units
            
        Raises:
            ValueError: If required state variables are missing
        """
        xp = self._array_module
        
        # Validate required inputs
        if 'temperature' not in state or 'pressure' not in state:
            raise ValueError(
                "State dictionary must contain 'temperature' and 'pressure' fields"
            )
        
        temperature = state['temperature']
        pressure = state['pressure']
        
        # Compute air density
        rho_air = self._compute_air_density(temperature, pressure)
        
        # Initialize total reflectivity (in linear units)
        z_total = xp.zeros_like(temperature)
        
        # Add rain contribution
        if self.include_rain and 'qr' in state:
            qr = state['qr']
            if self.rain_scheme == 'marshall_palmer':
                z_rain = self._reflectivity_rain_marshall_palmer(qr, rho_air)
            else:
                warnings.warn(f"Unknown rain scheme: {self.rain_scheme}. Using Marshall-Palmer.")
                z_rain = self._reflectivity_rain_marshall_palmer(qr, rho_air)
            z_total = z_total + z_rain
        
        # Add snow contribution
        if self.include_snow and 'qs' in state:
            qs = state['qs']
            if self.snow_scheme == 'sekhon_srivastava':
                z_snow = self._reflectivity_snow_sekhon_srivastava(qs, rho_air)
            else:
                warnings.warn(f"Unknown snow scheme: {self.snow_scheme}. Using Sekhon-Srivastava.")
                z_snow = self._reflectivity_snow_sekhon_srivastava(qs, rho_air)
            z_total = z_total + z_snow
        
        # Add graupel contribution
        if self.include_graupel and 'qg' in state:
            qg = state['qg']
            z_graupel = self._reflectivity_graupel(qg, rho_air)
            z_total = z_total + z_graupel
        
        # Add cloud water contribution (typically small)
        if self.include_cloud and 'qc' in state:
            qc = state['qc']
            z_cloud = self._reflectivity_cloud(qc, rho_air)
            z_total = z_total + z_cloud
        
        # Convert to dBZ if requested
        if return_dbz:
            return self._linear_to_dbz(z_total)
        else:
            return z_total
    
    def __call__(self, state: dict, return_dbz: bool = True) -> Union[np.ndarray, Array]:
        """Make the operator callable.
        
        This allows using the operator as: obs = operator(state)
        
        Args:
            state: Dictionary containing model state variables
            return_dbz: If True, return dBZ; if False, return linear units
            
        Returns:
            Simulated radar reflectivity
        """
        return self.forward(state, return_dbz=return_dbz)


def create_radar_operator(
    wavelength: float = 0.10,
    use_jax: bool = False,
    **kwargs
) -> RadarReflectivityOperator:
    """Convenience function to create a radar reflectivity operator.
    
    Args:
        wavelength: Radar wavelength in meters. Common values:
            - 0.10 m (S-band)
            - 0.05 m (C-band)
            - 0.03 m (X-band)
        use_jax: Use JAX arrays instead of NumPy
        **kwargs: Additional keyword arguments passed to RadarReflectivityOperator
        
    Returns:
        Configured RadarReflectivityOperator instance
        
    Example:
        >>> operator = create_radar_operator(wavelength=0.10)
        >>> state = {
        ...     'temperature': temperature_array,
        ...     'pressure': pressure_array,
        ...     'qr': rain_mixing_ratio,
        ...     'qs': snow_mixing_ratio
        ... }
        >>> reflectivity_dbz = operator(state)
    """
    return RadarReflectivityOperator(
        wavelength=wavelength,
        use_jax=use_jax,
        **kwargs
    )

