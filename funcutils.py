from jax import jit, vmap, jacfwd
import jax.numpy as jnp
import numpy as np

import jax
import jax_cosmo.scipy.interpolate as jci


def mapper(foo):
    def mapped(a, b):
        result = jax.vmap(lambda x: jax.vmap(lambda y: foo(x, y))(b))(a)
        return result
    return jax.jit(mapped) 



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


def get_unit_tangent(f):
  func = get_tangent(f)
  def ut(x):
    t = func(x)
    return t/jnp.linalg.norm(t)
  return jit(ut)

def get_second(f):
  return get_tangent(get_tangent(f))

def get_unit_nornal(f):
  return jit(get_unit_tangent(get_unit_tangent(f)))



def frenet_frame(f):
  tf = get_unit_tangent(f)
  tf2 = get_unit_nornal(f)
  def frenet(t):
    x = tf(t)
    y = tf2(t)
    tt3 = (jnp.cross(x, y))
    return jnp.array([x, y, tt3])
  return jit(frenet)

def make_circ(f, eps=0.1):
    fren = frenet_frame(f)
    def circ(t, s):  # Add `s` as an explicit argument
        theframe = fren(t)
        normal = theframe[1]
        binormal = theframe[2]
        # Directly compute the value for given `t` and `s`
        return f(t) + eps * jnp.cos(s) * normal + eps *jnp.sin(s) * binormal
    return jit(circ)  # 

def make_tube_surf(f, eps=0.1):
    circ = make_circ(f, eps)
    def tube_surf(t, s):
        return circ(t, s)  # Call circ directly
    return jit(tube_surf)

def make_discrete_tube_surf(f, num1, num2, a = 0, b = 1, eps=0.1):
  tf = make_tube_surf(f, eps)
  mtf = mapper(tf)
  return mtf(jnp.linspace(a, b, num1), jnp.linspace(0, 2 * jnp.pi, num2))

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

@jit
def average_distance(points):
  newpoints = jnp.concatenate([points, points[:1, :]])
  dists = newpoints[1:] - newpoints[:-1]
  return jnp.mean(jnp.linalg.norm(dists, axis=1))

@jit
def rescale_points(points):
  ad = average_distance(points)
  return points/ad

def transform_knot(func, gauge=0.01, a=0.0, b=1.0, init_points = 100):
  arcfunc = get_arclength(func, beg=a, end=b, num_points = init_points)
  totallength = arcfunc(b)
  num = jnp.round(totallength/gauge)
  newfunc = parametrized_by_arclength(func, beg=a, end=b, num_points=num)
  return newfunc, 0, float(totallength), int(num)

  