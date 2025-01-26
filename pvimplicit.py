import numpy as np
import pyvista as pv

def torus(x, y, z, R, r):
    """Implicit equation of a torus"""
    return ((x**2 + y**2 + z**2 + R**2 - r**2)**2 - 4*(R**2)*(x**2 + y**2))

def do_values(func, grid, name='implicit_values'):
    """Evaluate an implicit function on a grid"""
    # Create coordinates for evaluation
    x, y, z = grid.points.T

    # Evaluate the function
    values = func(x, y, z)

    # Add the scalar values to the grid
    grid[name] = values

    return grid

def make_grid(dims, spacing, origin):
    """Create a grid with the given dimensions, spacing, and origin"""
    # Create a grid to evaluate the implicit function
    grid = pv.ImageData(
        dimensions=dims,
        spacing=spacing,
        origin=origin
    )
    return grid


def do_surface(grid, level_set=0):
    """Create a surface mesh from a grid and level set"""
    # Create the isosurface at the zero level set
    surface = grid.contour([level_set])
    return surface

def plot_surface(surface, plotter):
    """Add a surface to a plotter"""
    # Create a plotter and add the surface
    plotter.add_mesh(surface, color='lightblue', smooth_shading=True)
    
    plotter.camera_position = 'xz'
    plotter.camera.zoom(1.5)

    # Show the plot
    plotter.show()

def do_all(func, dims, spacing, origin, level_set=0):
    """Evaluate an implicit function on a grid and create a surface mesh"""
    # Create a grid to evaluate the implicit function
    grid = make_grid(dims, spacing, origin)

    
    grid = do_values(func, grid)

    # Create the isosurface at the zero level set
    surface = do_surface(grid, level_set=level_set)

    # Create a plotter and add the surface
    plotter = pv.Plotter()
    plot_surface(surface, plotter)
    return surface
