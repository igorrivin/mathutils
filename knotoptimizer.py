import numpy as np
import jax
import jax.numpy as jnp 
from dataclasses import dataclass
from jax import jit, vmap
from jax import lax
import optax
from funcutils import rescale_points, transform_knot
import faiss
from jax import tree_util
from graphutils import make_edges
from knnutils import get_knn_indices

def get_faiss_indices(points, k=10):
    howmany = points.shape[0]
    k = min(k, howmany)
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
    def __hash__(self):
        return hash(tuple(vars(self).values()))

tree_util.register_pytree_node(
    Physics,
    lambda obj: ((), (obj.spring_constant, obj.repulsion_strength, obj.epsilon)),
    lambda aux, _: Physics(*aux)
)

# def pairwise_distance(points, i, j):
#     diff = points[i] - points[j]
#     return jnp.sqrt(jnp.sum(diff ** 2))



def make_streamlined(e):
    def pairwise_potential(points, edge):
        diff = points[edge[0]] - points[edge[1]]
        return jnp.sqrt(jnp.sum(diff ** 2)-1.0) ** 2

    def thepotential(points):
      potarray = jax.vmap(pairwise_potential, in_axes=(None, 0))(points, e)
      return jnp.mean(potarray)
    return jax.jit(thepotential)

# def make_streamlined(e):
#     def pairwise_distance(points, edge):
#         diff = points[edge[0]] - points[edge[1]]
#         return jnp.sqrt(jnp.sum(diff ** 2))

#     return jax.jit(jax.vmap(lambda points: jnp.mean(
#         (jax.vmap(pairwise_distance, in_axes=(None, 1))(points, e) - 1.0) ** 2)
#     ))


def make_potential(graph):
    e = make_edges(graph)
    return make_streamlined(e)

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
    


# @jit
# def repulsion_potential(points, physics : Physics):
#     """Compute repulsion potential between all pairs of points"""
#     n = points.shape[0]

#     # Compute pairwise distances using broadcasting
#     diffs = jnp.expand_dims(points, axis=0) - jnp.expand_dims(points, axis=1)  # n x n x 3
#     distances_squared = jnp.sum(diffs * diffs, axis=2)  # n x n
#     distances = jnp.sqrt(distances_squared)
#     # Create mask for self-interactions
#     mask = ~jnp.eye(n, dtype=bool)

#     # Compute repulsion (inverse square law)
#     # Add small epsilon to avoid division by zero
#     repulsion = mask / (distances + 1e-6)

#     # Return normalized repulsion potential
#     return physics.repulsion_strength * jnp.sum(repulsion) / (n * (n - 1))

from functools import partial

@partial(jit, static_argnames=[ "physics"])

def repulsion_potential(points, physics):
    n = points.shape[0]

    def body(i, val):
        # Compute pairwise differences for all points
        diffs = points - points[i]  # Broadcast points[i] to all rows
        distances = jnp.sqrt(jnp.sum(diffs**2, axis=1))  # Compute distances

        # Mask out invalid distances (i.e., j <= i)
        mask = jnp.arange(n) > i
        masked_distances = jnp.where(mask, distances, jnp.inf)  # Use `inf` to ignore

        # Compute potentials
        potentials = 1 / (masked_distances + physics.epsilon)
        return val + jnp.sum(potentials)

    # Use fori_loop for efficiency
    total_potential = 2.0 * physics.repulsion_strength * lax.fori_loop(0, n, body, 0.0) / (n * (n - 1))
    return total_potential
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

# Register OptimizationState as a PyTree
tree_util.register_pytree_node(
    OptimizationState,
    # Flatten function
    lambda obj: ((obj.points, obj.opt_state), ()),
    # Unflatten function
    lambda aux, children: OptimizationState(*children))

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
        #return  Optimizer.linear_potential(points)

    @staticmethod
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
            print(loss, grads)
            updates, opt_state = optimizer.update(grads, state.opt_state)
            points = optax.apply_updates(state.points, updates)

            return OptimizationState(points=points, opt_state=opt_state), (loss, points)

        # Use lax.scan for optimization steps
         # Use lax.scan for optimization steps
        state, loss_history = lax.scan(step, state, xs=None, length=num_steps)
        return state, loss_history
    
    @staticmethod
    def create_epoch_loss_function(initial_points, physics: Physics, num_neighbors = 10):
        """Create a function that computes the loss for a given set of points."""
        epsilon = physics.epsilon
        #index_pairs = jnp.array(get_faiss_indices(np.asarray(initial_points)))
        index_pairs = jnp.array(get_knn_indices(np.asarray(initial_points), k=num_neighbors))
        @jit
        def epoch_loss(points):
            def potential(i, j):
                return 1 / (jnp.linalg.norm(points[i] - points[j]) + epsilon)
            
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

def optimize_knot(func, *, linear_potential = spring_potential, sparse = False, gauge=0.01, a=0.0, b=1.0, init_points = 100, spring_constant=1.0, repulsion_strength=10.0, learning_rate=0.01, num_steps=100, num_epochs = 100):
  #newfunc, a, b, num = transform_knot(func, gauge, a, b, init_points)
  newfunc = func
  num = init_points
  knotpoints = vmap(newfunc)(jnp.linspace(a, b, num))
  Optimizer.linear_potential = linear_potential
  loss_list = [ ]
  for _ in range(num_epochs):
    state, physics, optimizer = make_optimizer(knotpoints, spring_constant=spring_constant, repulsion_strength=repulsion_strength, learning_rate=learning_rate)
    if sparse:
      state, loss_history = Optimizer.run_sparse_epoch(state, physics, optimizer, num_steps=num_steps)
    else:
      state, loss_history = Optimizer.run_epoch(state, physics, optimizer, num_steps=num_steps)
    print(loss_history[-1])
    knotpoints = state.points
    loss_list.append(loss_history)
  loss_histories = jnp.concatenate([x[0] for x in loss_list])
  trajectory = jnp.concatenate([x[1] for x in loss_list])
  return knotpoints, (loss_histories, trajectory)
  

def surface_projection(func, *, num_points = 1000, num_steps = 1000, spring_constant=1.0, repulsion_strength=0.001, learning_rate=0.01 ):
    newfunc = make_surface_potential(func)
    Optimizer.linear_potential = newfunc
    state, physics, optimizer = make_optimizer(newfunc, spring_constant=spring_constant, repulsion_strength=repulsion_strength, learning_rate=learning_rate)
    state, loss_history = Optimizer.run_sparse_epoch(state, physics, optimizer, num_steps=num_steps)
    return state.points, loss_history


def optimize_graph(graph, *, num_epochs = 10, num_steps = 100, spring_constant=1.0, repulsion_strength=0.1, learning_rate = 0.01):
    Optimizer.linear_potential = make_potential(graph)
    num_nodes = len(graph)
    points = jnp.array(np.random.rand(num_nodes, 3))
    loss_list = []
    
    for _ in range(num_epochs):
        state, physics, optimizer = make_optimizer(points, spring_constant=spring_constant, repulsion_strength=repulsion_strength, learning_rate=learning_rate)
        state, loss_history = Optimizer.run_sparse_epoch(state, physics, optimizer, num_steps=num_steps)
        loss_list.append(loss_history)
        points = state.points
    loss_histories = jnp.concatenate([x[0] for x in loss_list])
    trajectory = jnp.concatenate([x[1] for x in loss_list])
    return points, (loss_histories, trajectory)

def stupid_umap(points, num_neighbors = 10, num_epochs = 10, num_steps = 100, spring_constant=1.0, repulsion_strength=0.1, learning_rate = 0.01):
    index_pairs = get_knn_indices(points, k=num_neighbors)
    Optimizer.linear_potential = make_streamlined(index_pairs)
    num_nodes = points.shape[0]
    points = jnp.array(np.random.rand(num_nodes, 2))
    loss_list = []
    
    for i in range(num_epochs):
        print(i)
        state, physics, optimizer = make_optimizer(points, spring_constant=spring_constant, repulsion_strength=repulsion_strength, learning_rate=learning_rate)
        state, loss_history = Optimizer.run_sparse_epoch(state, physics, optimizer, num_steps=num_steps)
        loss_list.append(loss_history)
        points = state.points
    return points