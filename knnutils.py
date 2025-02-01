import numpy as np
import faiss
import jax.numpy as jnp
import jax.dlpack
import torch

try:
    import cupy as cp
    #from cuvs.nearest_neighbors import knn
    from cuvs.neighbors import cagra
    import cudf  # Only needed for optional dataframe conversion
    has_cuda = cp.cuda.runtime.getDeviceCount() > 0
except ImportError:
    has_cuda = False

print(f"Using CUDA: {has_cuda}")

def get_knn_indices(points, k=10):
    howmany = points.shape[0]
    k = min(k, howmany)

    def mung_neighbors(I):
        I = I[:, 1:]  # Remove self-adjacency
        print(I.shape)
        num_queries, k = I.shape
        row_indices = jnp.arange(num_queries).reshape(-1, 1)  # Column vector
        row_indices = jnp.broadcast_to(row_indices, (num_queries, k))  # Expand
        row_indices = row_indices.ravel()  # Flatten

        col_indices = I.ravel()  # Flatten neighbors
        index_pairs = jnp.column_stack((row_indices, col_indices)) 
        return index_pairs
    
    if has_cuda:
        # Ensure `points` is CuPy array (convert from JAX if needed)
        if isinstance(points, jnp.ndarray):
            points_cp = cp.from_dlpack(jax.dlpack.to_dlpack(points))
        else:
            points_cp = cp.asarray(points)

        # Compute k-nearest neighbors using cuVS (directly on CuPy)
        #result = knn(points_cp, points_cp, k + 1)  # k+1 to handle self-adjacency

        build_params = cagra.IndexParams(metric="sqeuclidean")
        index = cagra.build(build_params, points_cp)
        distances, neighbors = cagra.search(cagra.SearchParams(),
                                     index, points_cp,
                                     k)
        #distances = cp.asarray(distances)
        #I = cp.asarray(neighbors)
        I = jax.dlpack.from_dlpack(cp.asarray(neighbors))
        D = jax.dlpack.from_dlpack(cp.asarray(distances))
        I = I[:, 1:]  # Remove self-adjacency
        D = D[:, 1:]  # Remove self-adjacency
        return I, D
        # Extract adjacency list as CuPy arrays (faster than cudf.DataFrame)
        #source = result["source"].values  # CuPy array
        #destination = result["destination"].values  # CuPy array

        # Remove self-adjacency
        #mask = source != destination

        # Efficient stacking instead of column_stack
        #index_pairs = cp.vstack((source[mask], destination[mask])).T  # Faster memory access

        # Convert CuPy array to JAX array without copying to CPU
        #return jax.dlpack.from_dlpack(index_pairs.toDlpack())

    else:
        # Fallback: FAISS (Try GPU version first)
        use_gpu = faiss.get_num_gpus() > 0
        if use_gpu:
            res = faiss.StandardGpuResources()  # Use FAISS GPU resources
            index = faiss.GpuIndexFlatL2(res, points.shape[1])  # GPU FAISS
        else:
            index = faiss.IndexFlatL2(points.shape[1])  # CPU FAISS

        index.add(points)
        D, I = index.search(points, k + 1)  # k+1 to remove self-adjacency
        I = I[:, 1:]  # Remove self-adjacency
        D = I[:, 1:]
        return jnp.array(I), jnp.array(D)
        # Return as JAX array (ensures compatibility)
        return jnp.asarray(index_pairs)

import jax.random as random

# def generate_random_adj_list(num_points, k, key=random.PRNGKey(42)):
#     """Generates a (num_points, k) adjacency list where:
#        - Each row contains k unique neighbors.
#        - No row contains its own index.
#     """
#     keys = random.split(key, num_points)  # Unique PRNG keys for each row
#     row_indices = jnp.arange(num_points)  # ✅ Compute once outside vmap

#     def sample_neighbors(i, key):
#         """Samples k unique neighbors for row i, ensuring i is excluded."""
#         offsets = random.choice(key, jnp.arange(1, num_points), shape=(k,), replace=False)  # ✅ Sample from {1, ..., n-1}
#         neighbors = (i + offsets) % num_points  # ✅ Shift by i and take modulo num_points
#         return neighbors

#     adj_list = jax.vmap(sample_neighbors, in_axes = (0, 0))(row_indices, keys)  # ✅ Parallelized sampling
#     return adj_list

# # Example usage
# num_points = 100
# k = 5
# key = random.PRNGKey(42)

# adj_list = generate_random_adj_list(num_points, k, key)
# print(adj_list)

# # Example usage
# num_points = 100
# k = 5
# key = random.PRNGKey(42)

# adj_list = generate_random_adj_list(num_points, k, key)
# print(adj_list)

# # Example usage
# if __name__ == "__main__":
#     points = jnp.array(np.random.rand(100, 128).astype(np.float32))  # JAX-compatible input
#     knn_result = get_knn_indices(points)
#     print(knn_result.shape)



def generate_random_adj_list(num_points, k, key):
    """Generates a (num_points, k) adjacency list where:
       - Each row contains k random neighbors.
       - No row contains its own index (i.e., no self-loops).
    """
    rand_offsets = random.randint(key, shape=(num_points, k), minval=1, maxval=num_points)  # ✅ Random offsets (1 to num_points-1)
    row_indices = jnp.arange(num_points).reshape(-1, 1)  # ✅ Shape (num_points, 1)
    neighbors = (row_indices + rand_offsets) % num_points  # ✅ Ensures no self-loops
    return neighbors

# Example usage
num_points = 100
k = 5
key = random.PRNGKey(42)

adj_list = generate_random_adj_list(num_points, k, key)
print(adj_list)