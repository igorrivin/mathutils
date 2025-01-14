from jax import jit, vmap, jacfwd
import jax.numpy as jnp
import numpy as np

import jax
import jax_cosmo.scipy.interpolate as jci


@jit
def traprulej(x, y):
  ar1 = x[1:] * y[1:]
  ar2 = y[1:] * x[:-1]
  ar3 = x[:-1] * y[:-1]
  ar4 = y[:-1] * x[1:]
  
  ar = jnp.cumsum(ar1 - ar2 - ar3 + ar4)
  return jnp.concatenate([np.array([0]), ar])/2

@jit
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
  return traprulej(knots, jax.vmap(cfunc)(knots))[-1]

def get_num(f, gauge, beg=0.0, end = 1.0, num_points = 100):
  arcfunc = get_arclength(f, beg=beg, end=end, num_points = num_points)
  totallength = arcfunc(end)
  return jnp.round(totallength/gauge)