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
        I = np.asarray(neighbors)
        I = I[:, 1:]  # Remove self-adjacency
        print(I.shape)
        num_queries, k = I.shape
        row_indices = np.repeat(np.arange(num_queries), k)
        col_indices = I.ravel()
        index_pairs = np.column_stack((row_indices, col_indices))

        # Return as JAX array (ensures compatibility)
        return jnp.asarray(index_pairs)
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
        num_queries, k = I.shape
        row_indices = np.repeat(np.arange(num_queries), k)
        col_indices = I.ravel()
        index_pairs = np.column_stack((row_indices, col_indices))

        # Return as JAX array (ensures compatibility)
        return jnp.asarray(index_pairs)


# Example usage
if __name__ == "__main__":
    points = jnp.array(np.random.rand(100, 128).astype(np.float32))  # JAX-compatible input
    knn_result = get_knn_indices(points)
    print(knn_result.shape)