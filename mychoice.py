import jax.numpy as jnp
import jax.random as random
from jax import lax, jit, vmap
from functools import partial

def swap_elements(arr, i, j):
    temp = arr[j]  # Store arr[j] before modifying it
    arr = arr.at[j].set(arr[i])
    arr = arr.at[i].set(temp)
    return arr

arr = jnp.array([1, 2, 3, 4, 5])
new_arr = swap_elements(arr, 1, 3)
print(new_arr)  # Output: [1, 4, 3, 2, 5]
@partial(jit, static_argnums=(0, 1))
def mychoice(num, k, key=random.PRNGKey(42)):
    ar = jnp.arange(num)
    def body_func(i, state):
        newkey = random.fold_in(key, i)
        myrand = random.randint(newkey, shape=(), minval=i, maxval=num)
        return swap_elements(state, i, myrand)
    return lax.fori_loop(
        lower=0,
        upper=k,
        body_fun=body_func,
        init_val=ar
    )[:k]

@partial(jit, static_argnums=(0, 1, 2))
def mychoice_batched(num, k, batch_size, key=random.PRNGKey(42)):
    num_full_batches = num // batch_size  # Only the full batches
    
    def process_batch(batch_idx, results):
        batch_key = random.fold_in(key, batch_idx)
        start_idx = batch_idx * batch_size
        # Now we're always processing full batches
        batch_results = vmap(
            lambda batch_key: mychoice(num, k, batch_key)
        )(random.split(batch_key, batch_size))
        
        return results.at[start_idx:start_idx + batch_size].set(batch_results)

    results = lax.fori_loop(
        lower=0,
        upper=num_full_batches,
        body_fun=process_batch,
        init_val=jnp.zeros((num, k), dtype=jnp.int32)
    )
    
    # Handle the remainder separately if needed
    remainder = num % batch_size
    if remainder > 0:
        last_key = random.fold_in(key, num_full_batches)
        start_idx = num_full_batches * batch_size
        last_results = vmap(
            lambda batch_key: mychoice(num, k, batch_key)
        )(random.split(last_key, remainder))
        results = results.at[start_idx:num].set(last_results)
    
    return results

@partial(jit, static_argnums=(0, 1, 2))  # Make num, k, and batch_size static
def builtin_choice_batched(num, k, batch_size, key=random.PRNGKey(42)):
    num_batches = (num + batch_size - 1) // batch_size
    
    def process_batch(batch_idx, results):
        batch_key = random.fold_in(key, batch_idx)
        start_idx = batch_idx * batch_size
        end_idx = jnp.minimum(start_idx + batch_size, num)
        
        batch_results = vmap(
            lambda batch_key: random.choice(batch_key, jnp.arange(num), shape=(k,), replace=False)
        )(random.split(batch_key, end_idx - start_idx))
        
        return results.at[start_idx:end_idx].set(batch_results)

    return lax.fori_loop(
        lower=0,
        upper=num_batches,
        body_fun=process_batch,
        init_val=jnp.zeros((num, k), dtype=jnp.int32)
    )

@partial(jit, static_argnums=(0, 1))
def mychoice_fast(num, k, key=random.PRNGKey(42)):
    # Initialize reservoir with first k numbers
    result = jnp.arange(k, dtype=jnp.int32)
    
    # Initialize W = exp(log(random)/k)
    key1, key2 = random.split(key)
    W = random.uniform(key1, dtype=jnp.float32) ** (1.0/k)
    
    def body_fun(state):
        i, W, key, result = state
        
        key1, key2, key3, key = random.split(key, 4)
        
        # Make sure all computations are in float32
        skip = jnp.floor(
            jnp.log(random.uniform(key1, dtype=jnp.float32)) / 
            jnp.log(1.0 - W)
        ).astype(jnp.int32) + 1
        
        new_i = i + skip
        idx = random.randint(key2, shape=(), minval=0, maxval=k)
        new_W = W * (random.uniform(key3, dtype=jnp.float32) ** (1.0/k))
        
        result = jnp.where(new_i < num, 
                          result.at[idx].set(new_i), 
                          result)
        
        return new_i, new_W, key, result
    
    def cond_fun(state):
        i, W, key, result = state
        return i < num
    
    final_i, final_W, final_key, final_result = lax.while_loop(
        cond_fun,
        body_fun,
        (k, W, key2, result)
    )
    
    return final_result

def profile_all_versions(num, k, batch_size, num_trials=10):
    key = random.PRNGKey(0)
    
    # Warm up JIT
    _ = mychoice_batched(num, k, batch_size, key)
    _ = builtin_choice_batched(num, k, batch_size, key)
    _ = vmap(lambda k: random.choice(key, jnp.arange(num), shape=(k,), replace=False))(jnp.ones(num))
    
    # Time custom batched version
    start = time.time()
    for _ in range(num_trials):
        result1 = mychoice_batched(num, k, batch_size, key)
    custom_batch_time = (time.time() - start) / num_trials
    
    # Time builtin batched version
    start = time.time()
    for _ in range(num_trials):
        result2 = builtin_choice_batched(num, k, batch_size, key)
    builtin_batch_time = (time.time() - start) / num_trials
    
    # Time standard vmapped version
    start = time.time()
    for _ in range(num_trials):
        result3 = vmap(lambda k: random.choice(key, jnp.arange(num), shape=(k,), replace=False))(jnp.ones(num))
    standard_time = (time.time() - start) / num_trials
    
    print(f"Average time over {num_trials} trials:")
    print(f"Custom batched implementation: {custom_batch_time:.4f} seconds")
    print(f"Builtin batched implementation: {builtin_batch_time:.4f} seconds")
    print(f"Standard vmapped implementation: {standard_time:.4f} seconds")
    
    print(f"\nMemory usage:")
    print(f"Custom batched implementation: {result1.nbytes / 1024**2:.2f} MB")
    print(f"Builtin batched implementation: {result2.nbytes / 1024**2:.2f} MB")
    print(f"Standard vmapped implementation: {result3.nbytes / 1024**2:.2f} MB")
    
    return result1, result2, result3

