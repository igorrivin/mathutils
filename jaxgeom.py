from jax import grad, jit, vmap, jacfwd
from jax import random
from functools import partial
import jax.numpy as jnp
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from funcutils import transform_knot

def get_uvec2(f):
  tanvec = jit(jacfwd(f))
  return lambda x: tanvec(x)/jnp.linalg.norm(tanvec(x))

def get_cvec(f):
  return get_uvec2(get_uvec2(f))

def get_cvec2(tt):
  return get_uvec2(tt)

""" def traprulej(x, y):
  ar1 = x[1:] * y[1:]
  ar2 = y[1:] * x[:-1]
  ar3 = x[:-1] * y[:-1]
  ar4 = y[:-1] * x[1:]
  
  ar = jnp.cumsum(ar1 - ar2 - ar3 + ar4)
  return jnp.concatenate([np.array([0]), ar])/2

def do_inverse(func, beg=0.1, end = 1.0, num_points =100):
  xes = jnp.linspace(beg, end, num_points)
  ys = vmap(func)(xes)
  interp = jci.InterpolatedUnivariateSpline(ys, xes)
  return jit(interp)

def get_tangent(f):
   # return a function which evaluates the tangent at its argument.
   return jit(jacfwd(f))

def get_second(f):
  return get_tangent(get_tangent(f))

def get_curvature(f):
  tf = get_tangent(f)
  tf2 = get_second(f)
  def cfunc(x):
    t = get_tangent(f)(x)
    n = get_second(f)(x)
    tnorm = jnp.linalg.norm(t)
    nnorm = jnp.linalg.norm(n)
    return jnp.sqrt(tnorm*tnorm * nnorm * nnorm - (jnp.dot(t, n))**2)/tnorm**3
  return jit(cfunc)

def get_arclength_d(f):
  def foo(x):
    v = get_tangent(f)(x)
    return jnp.linalg.norm(v)
  return jit(foo)

def get_arclength(f, beg=0.0, end=1.0, num_points = 100):
  deriv = get_arclength_d(f)
  xvals = jnp.linspace(beg, end, num_points)
  yvals = vmap(deriv)(xvals)
  intvals = traprulej(xvals, yvals)
  interp = jci.InterpolatedUnivariateSpline(xvals, intvals)
  return jit(interp)

def get_arclength_inv(f, beg=0.0, end = 1.0, num_points = 100):
  return jit(do_inverse(get_arclength(f, beg=beg, end=end, num_points = num_points)))

def parametrized_by_arclength(f, beg=0.0, end = 1.0, num_points = 100):
  def thefunc(x):
    return f(get_arclength_inv(f, beg=beg, end=end, num_points = num_points)(x))
  return jit(thefunc)

def get_total_curvature(f, a, b, num_points = 100):
  cfunc = get_curvature(f)
  knots = jnp.linspace(a, b, num_points)
  return traprulej(knots, vmap(cfunc)(knots))[-1]
 """
def get_frame(f):
  tt = get_uvec2(f)
  tt2 = get_cvec2(tt)
  def first2(t):
    x = tt(t)
    y = tt2(t)
    tt3 = (jnp.cross(x, y))
    return jnp.array([x, y, tt3])
  return jit(first2)

def get_point(frame, s):
  v1 = frame[1, :]
  v2 = frame[2, :]
  return jnp.cos(s) * v1 + jnp.sin(s) * v2

def get_grid(f, eps):
  ffunc = get_frame(f)
  def grid(t, s):
    base = f(t)
    frame = ffunc(t)
    return base + eps * get_point(frame, s)
  return grid

def get_grida(f, eps):
  ffunc = get_frame(f)
  def grid(ar):
    t = ar[0]
    s = ar[1]
    base = f(t)
    frame = ffunc(t)
    return base + eps * get_point(frame, s)
  return grid

def get_gridb(f, eps):
  ffunc = get_frame(f)
  def grid(ar):
    t = ar[0]
    s = ar[1]
    r = jnp.sqrt(ar[2])
    base = f(t)
    frame = ffunc(t)
    return base + eps * r * get_point(frame, s)
  return grid

#@partial(jit, static_argnames=['f', ])
def get_reg_grid(f, num1, num2, l,eps):
  tarray = jnp.linspace(start = 0.0, stop = l, num = num1)
  sarray = jnp.linspace(start = 0.0, stop = 2 * jnp.pi, num = num2)
  g = get_grid(f, eps)
  g = vmap(g, in_axes=(None, 0))
  g = vmap(g, in_axes=(0, None))
  return jnp.vstack(g(tarray, sarray))


@partial(jit, static_argnames=['f', 'num'])
def get_irreg_grid(f, num, eps):
  parray = jnp.array(np.random.rand(num, 2) * np.array([1, 2 * np.pi]))
  g = get_grida(f, eps)
  g = vmap(g, in_axes=( 0))
  return jnp.vstack(g(parray))

@partial(jit, static_argnames=['f', 'num'])
def get_irreg_grid_full(f, num, eps):
  parray = jnp.array(np.random.rand(num, 3) * np.array([1, 2 * np.pi, 1]))
  g = get_gridb(f, eps)
  g = vmap(g, in_axes=( 0))
  return jnp.vstack(g(parray))

def make_rand_knot(decay, l):
  #seed = int(time.time())
  seed = np.random.randint(0, 1e6)
  key = jr.PRNGKey(seed)
  key, subkey = jr.split(key)
  thebatch1x = jr.normal(subkey, shape=(l,))
  key, subkey = jr.split(key)
  thebatch2x = jr.normal(subkey, shape=(l,))
  key, subkey = jr.split(key)
  thebatch1y = jr.normal(subkey, shape=(l,))
  key, subkey = jr.split(key)
  thebatch2y= jr.normal(subkey, shape=(l,))
  key, subkey = jr.split(key)
  thebatch1z = jr.normal(subkey, shape=(l,))
  key, subkey = jr.split(key)
  thebatch2z= jr.normal(subkey, shape=(l,))
  scaler = jnp.arange(1, l+1, dtype = jnp.float32)**(-decay)
  coefs1x = thebatch1x * scaler
  coefs2x = thebatch2x * scaler
  coefs1y = thebatch1y * scaler
  coefs2y = thebatch2y * scaler
  coefs1z = thebatch1z * scaler
  coefs2z = thebatch2z * scaler
  def make_ser(x):
    c = jnp.arange(1, l+1, dtype = jnp.float32)
    cc = c * 2 * jnp.pi * x
    coses = jnp.cos(cc)
    sines = jnp.sin(cc)
    return jnp.array([jnp.sum(coefs1x * coses) + jnp.sum(coefs2x * sines), jnp.sum(coefs1y * coses) + jnp.sum(coefs2y * sines), jnp.sum(coefs1z * coses) + jnp.sum(coefs2z * sines)])
  return make_ser

def make_trefoil():
  tp = 2 * jnp.pi
  return lambda t: jnp.array([jnp.sin(tp * t) + 2 * jnp.sin(2 * tp*t), jnp.cos(tp*t) - 2 * jnp.cos(2 * tp*t), -jnp.sin(3*tp*t)])


def make_fig8():
  tp = 2 * jnp.pi
  return lambda t: jnp.array([(2+jnp.cos(2*tp*t))* jnp.cos(3*tp*t), (2 + jnp.cos(2*tp*t))*jnp.sin(3*tp*t), jnp.sin(4*tp*t)])

  
def get_knot(func, howmany = 1000):
  ff = vmap(func)
  args = jnp.linspace(0, 1, howmany)
  return(ff(args))

def grid_to_trimesh(vertices, num1, num2):
    """
    Convert a regular grid of vertices to a trimesh.
    
    Args:
        vertices: torch.Tensor of shape (N, 3) containing the vertex positions
        num1: number of vertices in the first dimension
        num2: number of vertices in the second dimension
    
    Returns:
        trimesh.Trimesh object
    """
    import trimesh
    
    # Convert vertices to numpy if they're torch tensors
    
    # Create faces by connecting vertices in a grid pattern
    faces = []
    for i in range(num1 - 1):
        for j in range(num2 - 1):
            # Get vertex indices for current quad
            v0 = i * num2 + j
            v1 = v0 + 1
            v2 = (i + 1) * num2 + j
            v3 = v2 + 1
            
            # Create two triangles for the quad
            faces.append([v0, v2, v1])  # First triangle
            faces.append([v1, v2, v3])  # Second triangle
    
    faces = np.array(faces)
    
    # Create the mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh

def grid2d_to_trimesh(vertices):
  verts = np.asarray(vertices)
  n1, n2 , _ = verts.shape
  return grid_to_trimesh(verts.reshape(n1*n2, 3), n1, n2)

def make_nice_grid(f, num1= 100, num2=20, gauge = 0.01, a=0.0, b=1.0, eps = 0.1):
  from funcutils import transform_knot, make_discrete_tube_surf
  newf, a, b, num = transform_knot(f, gauge, a, b, num1)
  newgrid = make_discrete_tube_surf(newf, a, b, num, num2, eps=eps)
