import numpy as np
import jax
import jax.numpy as jnp
from flax.struct import dataclass
from jax import jit
from jax import lax
import optax
from funcutils import rescale_points



@dataclass
class Physics:
    spring_constant: float = 1.0
    repulsion_strength: float = 10.0

@jit
def spring_potential(points, physics: Physics):
    """
    Compute spring potential between adjacent points with unit target length.
    """
    # Get differences for adjacent points
    diffs = points[1:] - points[:-1]
    
    # Add the difference between the last and the first point
    closing_diff = points[0] - points[-1]
    
    # Concatenate all differences
    diffs = jnp.concatenate([diffs, jnp.expand_dims(closing_diff, axis=0)], axis=0)
    
    # Compute distances (L2 norm along the last axis)
    distances = jnp.linalg.norm(diffs, axis=1)
    
    # Compute the spring potential using unit target length
    return physics.spring_constant * jnp.mean((distances - 1.0) ** 2)

@jit

def repulsion_potential(points, physics : Physics):
    """Compute repulsion potential between all pairs of points"""
    n = points.shape[0]

    # Compute pairwise distances using broadcasting
    diffs = jnp.expand_dims(points, axis=0) - jnp.expand_dims(points, axis=1)  # n x n x 3
    distances_squared = jnp.sum(diffs * diffs, axis=2)  # n x n

    # Create mask for self-interactions
    mask = ~jnp.eye(n, dtype=bool)

    # Compute repulsion (inverse square law)
    # Add small epsilon to avoid division by zero
    repulsion = mask / (distances_squared + 1e-6)

    # Return normalized repulsion potential
    return physics.repulsion_strength * jnp.sum(repulsion) / (n * (n - 1))

@jit
def loss_function(points, physics):
    """Sum of the spring potential and repulsion potential"""
    spring_pot = spring_potential(points, physics)
    repulsion_pot = repulsion_potential(points, physics)
    return spring_pot + repulsion_pot


@dataclass
class OptimizationState:
    points: jnp.ndarray
    opt_state: optax.OptState



def make_optimizer(
    points: jnp.array,
    spring_constant: float = 1.0,
    repulsion_strength: float = 10.0,
    learning_rate: float = 0.01,
):
    """Factory function to set up the optimizer."""

    points = rescale_points(points)
    # Define the optimizer
    optimizer = optax.adam(learning_rate)

    # Initialize physics parameters
    physics = Physics(
        spring_constant=spring_constant,
        repulsion_strength=repulsion_strength,
    )

    # Return bundled state and helper objects
    return OptimizationState(
        points=points,
        opt_state=optimizer.init(points),
    ), physics, optimizer

class Optimizer:
    @staticmethod
    @jax.jit
    def compute_loss(points, physics: Physics):
        """Example loss function."""
        return spring_potential(points, physics) + repulsion_potential(points, physics)

    @staticmethod
    def run_epoch(state, physics, optimizer, num_steps):
        """Run one epoch using lax.scan."""
        @jax.jit
        def step(state: OptimizationState, _):
            def loss_fn(points):
                return Optimizer.compute_loss(points, physics)

            loss, grads = jax.value_and_grad(loss_fn)(state.points)
            updates, opt_state = optimizer.update(grads, state.opt_state)
            points = optax.apply_updates(state.points, updates)

            return OptimizationState(points=points, opt_state=opt_state), (loss, points)

        # Use lax.scan for optimization steps
         # Use lax.scan for optimization steps
        state, loss_history = lax.scan(step, state, xs=None, length=num_steps)
        return state, loss_history
   
#Example usage:
#     state, physics, optimizer = make_optimizer(
#     tn,
#     spring_constant=1.0,
#     repulsion_strength=10.0,
#     learning_rate=0.001,
# )

#state, loss_history = Optimizer.run_epoch(state, physics, optimizer, num_steps=100)