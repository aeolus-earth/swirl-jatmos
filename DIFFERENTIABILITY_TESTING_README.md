# Differentiability Testing for Swirl-Jatmos

This directory contains tools and documentation for testing the differentiability of Swirl-Jatmos physics modules for data assimilation applications.

## Files

### 1. `.cursorrules`
Comprehensive project documentation and coding guidelines for Swirl-Jatmos development.

**Contents:**
- Project architecture overview
- Physics modules description
- Numerical methods
- Grid structure (staggered Arakawa C-grid)
- Naming conventions
- Data assimilation integration notes

**Use:** Reference for understanding the codebase structure and conventions.

### 2. `test_differentiability.py`
Automated test suite for checking JAX automatic differentiation through physics modules.

**Usage:**
```bash
# Run all tests (except full forward step)
python3 test_differentiability.py --notest_full_step

# Run with full forward step test (slower)
python3 test_differentiability.py --test_full_step

# Run with less verbose output
python3 test_differentiability.py --noverbose
```

**What it tests:**
- Thermodynamics (water equilibrium, phase transitions)
- Microphysics (autoconversion, precipitation)
- Convection/advection schemes
- Diffusion operators
- SGS turbulence model
- Boundary conditions
- Poisson solver structure

**Output:** Pass/fail for each module with gradient norms and error messages.

### 3. `DIFFERENTIABILITY_REPORT.md`
Comprehensive assessment report of differentiability testing results.

**Contents:**
- Executive summary (80% pass rate)
- Module-by-module analysis
- Identified issues and solutions
- Data assimilation recommendations
- Code examples for gradient computation
- Performance considerations

**Key Findings:**
- ✅ Most basic physics is differentiable
- ⚠️ Thermodynamic equilibrium solver needs array shape fixes
- ⚠️ Iterative Poisson solver may need implicit differentiation
- ✅ Boundary conditions are fully differentiable
- ✅ Advection, diffusion, SGS model are differentiable

## Quick Start: Testing Differentiability

### Prerequisites
```bash
# Install dependencies
pip install jax jaxlib numpy scipy dataclasses-json etils absl-py
```

### Run Tests
```bash
cd /path/to/swirl-jatmos
python3 test_differentiability.py
```

### Interpret Results
- **✓ PASS**: Module is differentiable, gradients are finite
- **✗ FAIL**: Module has differentiability issues
- **⚠ WARNING**: Module not tested or needs special consideration

## For Data Assimilation Users

### Can I use Swirl-Jatmos with gradient-based DA?
**Yes, with minor modifications.** 80% of tested modules are already differentiable.

### What needs to be fixed?
1. **Thermodynamic equilibrium solver** - Array shape compatibility
2. **Poisson solver** - Consider implicit differentiation for iterative methods
3. **Test radiation module** - RRTMGP not yet tested

### Example: Computing Gradients
```python
import jax
from swirl_jatmos import navier_stokes_step

def loss_function(state, cfg):
    """Example loss: predict observations."""
    next_state, aux = navier_stokes_step.step(state, poisson_solver, cfg)
    # Your observation operator here
    predicted_obs = observation_operator(next_state)
    return jnp.sum((predicted_obs - true_obs)**2)

# Compute gradient
grad_fn = jax.grad(loss_function)
gradient = grad_fn(initial_state, cfg)
```

### Recommended DA Methods

**4D-Var (Variational):**
- Feasibility: High (with fixes)
- Requires: Differentiable forward model ✓
- Use: Optimize initial conditions or parameters

**Parameter Estimation:**
- Feasibility: High
- Example: Tune SGS constants, microphysics parameters
- Use: Model calibration

**Ensemble Methods (EnKF, EnSRF):**
- Feasibility: Always possible
- Advantage: No differentiability required
- Disadvantage: Computationally expensive

## Common Issues and Solutions

### Issue 1: "vmap shape error"
**Symptom:** `vmap was requested to map its argument along axis 0...`

**Solution:** Ensure all array inputs have compatible shapes
```python
# Bad: p_ref is 1D
p_ref = jnp.array([1e5, 9e4, 8e4, ...])  # shape (nz,)

# Good: broadcast to 3D
p_ref_3d = p_ref[None, None, :] * jnp.ones((nx, ny, nz))
```

### Issue 2: "Module has no attribute"
**Symptom:** Function name not found

**Solution:** Check actual API in module, update test accordingly

### Issue 3: Zero or vanishing gradients
**Symptom:** Optimization doesn't converge

**Cause:** Clipping operations (`jnp.clip`) cause zero gradients outside range

**Solution:**
- Use soft thresholds: `jax.nn.softplus`
- Check gradient magnitudes
- Adjust optimization learning rate

## Performance Notes

### Memory Usage
Gradient computation requires storing intermediate activations:
- **Forward pass:** Base memory
- **Backward pass:** ~2-3× base memory

**Solution:** Use `jax.checkpoint` for long integrations
```python
@jax.checkpoint
def timestep(state):
    return navier_stokes_step.step(state, ...)
```

### Computation Time
- **Forward pass:** 1×
- **Gradient (backward pass):** ~2-3× forward
- **Total (forward + backward):** ~3-4× forward only

### Scaling
JAX's AD scales well:
- Tested up to millions of parameters
- Parallel gradient computation with `jax.pmap`

## Advanced Topics

### Implicit Differentiation for Iterative Solvers

For Poisson solver or other iterative methods:
```python
@jax.custom_vjp
def implicit_solve(A, b):
    """Solve Ax = b with implicit differentiation."""
    x = iterative_solver(A, b)
    return x

def solve_fwd(A, b):
    x = implicit_solve(A, b)
    return x, (A, x)

def solve_bwd(res, g):
    A, x = res
    # Adjoint: solve A^T λ = g
    lam = iterative_solver(A.T, g)
    return (-jnp.outer(lam, x), lam)

implicit_solve.defvjp(solve_fwd, solve_bwd)
```

### Gradient Validation

Always validate gradients with finite differences:
```python
def finite_difference_gradient(f, x, eps=1e-5):
    """Compute gradient via centered finite differences."""
    grad = jnp.zeros_like(x)
    for i in range(x.size):
        x_plus = x.at[i].add(eps)
        x_minus = x.at[i].add(-eps)
        grad = grad.at[i].set((f(x_plus) - f(x_minus)) / (2*eps))
    return grad

# Compare
jax_grad = jax.grad(loss_fn)(x)
fd_grad = finite_difference_gradient(loss_fn, x)
print("Gradient error:", jnp.linalg.norm(jax_grad - fd_grad))
```

### Sparse Sensitivity Analysis

Identify most sensitive variables:
```python
def sensitivity_analysis(state, obs):
    """Compute sensitivity of observations to each state variable."""
    grad = jax.grad(lambda s: loss(s, obs))(state)
    
    sensitivities = {}
    for key, g in grad.items():
        sensitivities[key] = jnp.linalg.norm(g)
    
    return sensitivities
```

## Future Work

### High Priority
1. ✅ Complete basic differentiability audit
2. 🔄 Fix thermodynamic equilibrium solver
3. ⏳ Test full forward model (multi-step integration)
4. ⏳ Implement implicit diff for Poisson solver

### Medium Priority
5. ⏳ Test RRTMGP radiation module
6. ⏳ Develop 4D-Var example application
7. ⏳ Benchmark gradient computation performance
8. ⏳ Create DA tutorial notebooks

### Low Priority
9. ⏳ Implement gradient checkpointing examples
10. ⏳ Add unit tests for adjoint validation
11. ⏳ Document gradient computation in user guide

## References

### Papers on Differentiable Weather/Climate Models
- Sandu et al. (2005) - "Adjoint sensitivity analysis of regional air quality models"
- Alexe et al. (2015) - "Carbon flux estimation using model-data fusion"
- Farchi et al. (2021) - "Using machine learning to correct model error in DA"

### JAX Documentation
- [JAX Autodiff Cookbook](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html)
- [Custom VJP/JVP](https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html)
- [Checkpointing](https://jax.readthedocs.io/en/latest/jax-101/04-advanced-autodiff.html)

### Data Assimilation Resources
- Asch et al. (2016) - "Data Assimilation: Methods, Algorithms, and Applications"
- Kalnay (2003) - "Atmospheric Modeling, Data Assimilation and Predictability"
- Evensen (2009) - "Data Assimilation: The Ensemble Kalman Filter"

## Support

For questions or issues:
1. Check this README and the report
2. Review test output for specific errors
3. Consult JAX documentation for AD issues
4. Open issue on Swirl-Jatmos GitHub

## Contributing

To add new tests:
1. Add test function to `test_differentiability.py`
2. Follow existing test pattern
3. Document expected behavior
4. Update this README

---

**Last Updated:** November 11, 2025  
**Maintainer:** Differentiability Testing Suite  
**License:** Apache 2.0 (same as Swirl-Jatmos)

