import numpy as np
import jax
import jax.numpy as jnp 
from flax.struct import dataclass
from jax import jit, vmap
from jax import lax
import optax
from funcutils import rescale_points, transform_knot
import faiss


def get_faiss_indices(points, k=10):
    # Create a flat index
    index = faiss.IndexFlatL2(points.shape[1])   
    # Add the points to the index
    index.add(points)
    D, I = index.search(points, k)
    I = I[:, 1:] #delete self-adjacency
    num_queries, k = I.shape
    row_indices = np.repeat(np.arange(num_queries), k)
    col_indices = I.ravel()
    index_pairs = np.column_stack((row_indices, col_indices))
    return index_pairs


@dataclass
class Physics:
    spring_constant: float = 1.0
    repulsion_strength: float = 10.0
    epsilon: float = 1e-6

@jit
def spring_potential(points):
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
    return jnp.mean((distances - 1.0) ** 2)


def make_surface_potential(func):
    def potential(xyz):
        fval = func(xyz[0], xyz[1], xyz[2])
        return fval ** 2
    def potential_func(points):
        return jnp.mean(vmap(potential, in_axes=[0])(points))
    
    return jit(potential_func)
    


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
        epsilon=1e-6,
    )

    # Return bundled state and helper objects
    return OptimizationState(
        points=points,
        opt_state=optimizer.init(points),
    ), physics, optimizer

class Optimizer:
    @staticmethod
    def linear_potential(points):
        return spring_potential(points)
    @staticmethod
    @jax.jit
    def compute_loss(points, physics: Physics):
        """Example loss function."""
        return physics.spring_constant * Optimizer.linear_potential(points) +  repulsion_potential(points, physics)

    @staticmethod
    @jax.jit
    def compute_sparse_loss(points, physics: Physics):
        """Example loss function."""
        return physics.spring_constant * Optimizer.linear_potential(points) + physics.repulsion_strength *Optimizer.compute_sparse_repulsion(points)
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
    
    @staticmethod
    def create_epoch_loss_function(initial_points, physics: Physics):
        """Create a function that computes the loss for a given set of points."""
        epsilon = physics.epsilon
        index_pairs = jnp.array(get_faiss_indices(np.asarray(initial_points)))
        @jit
        def epoch_loss(points):
            def potential(i, j):
                return 1 / (jnp.linalg.norm(points[i] - points[j]) ** 2+ epsilon)
            
            potentials = vmap(potential)(index_pairs[:, 0], index_pairs[:, 1])
            return jnp.sum(potentials)
        
        Optimizer.compute_sparse_repulsion = epoch_loss
    
    @staticmethod
    def compute_sparse_repulsion(points, physics: Physics):
        """Compute repulsion potential between all pairs of points"""
        raise NotImplementedError
    
    @staticmethod
    def run_sparse_epoch(state, physics, optimizer, num_steps):
        """Run one sparse epoch using lax.scan."""
        Optimizer.create_epoch_loss_function(state.points, physics)
        @jax.jit
        def step(state: OptimizationState, _):
            def loss_fn(points):
                return Optimizer.compute_sparse_loss(points, physics)

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

def optimize_knot(func, *, linear_potential = spring_potential, sparse = False, gauge=0.01, a=0.0, b=1.0, init_points = 100, spring_constant=1.0, repulsion_strength=10.0, learning_rate=0.01, num_steps=100):
  newfunc, a, b, num = transform_knot(func, gauge, a, b, init_points)
  knotpoints = vmap(newfunc)(jnp.linspace(a, b, num))
  Optimizer.linear_potential = linear_potential
  state, physics, optimizer = make_optimizer(knotpoints, spring_constant=spring_constant, repulsion_strength=repulsion_strength, learning_rate=learning_rate)
  if sparse:
    state, loss_history = Optimizer.run_sparse_epoch(state, physics, optimizer, num_steps=num_steps)
  else:
    state, loss_history = Optimizer.run_epoch(state, physics, optimizer, num_steps=num_steps)
  return state.points, loss_history