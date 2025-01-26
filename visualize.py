from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from IPython.display import HTML

def visualize_evolution(trajectory,save_int = 100, interval=50):
    """
    Animate the evolution of the knot.
    
    Args:
        trajectory: list of point tensors from KnotEvolver
        interval: time between frames in milliseconds
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML
    
    # Convert trajectory tensors to numpy arrays
    trajectory_np = [np.asarray(points) for points in trajectory]
    
    # Get bounds for consistent axes
    all_points = np.concatenate(trajectory_np)
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)
    
    # Create figure and 3D axes
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Initialize line
    line, = ax.plot([], [], [], 'b-', lw=2)
    
    # Set consistent axis limits
    margin = 0.1 * (max_vals - min_vals)
    ax.set_xlim(min_vals[0] - margin[0], max_vals[0] + margin[0])
    ax.set_ylim(min_vals[1] - margin[1], max_vals[1] + margin[1])
    ax.set_zlim(min_vals[2] - margin[2], max_vals[2] + margin[2])
    
    # Add labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Animation update function
    def update(frame):
        points = trajectory_np[frame]
        # Include the first point again at the end to close the loop
        points_closed = np.vstack([points, points[0]])
        line.set_data(points_closed[:, 0], points_closed[:, 1])
        line.set_3d_properties(points_closed[:, 2])
        ax.view_init(30, frame % 360)  # Rotate view
        ax.set_title(f'Step {frame * save_int}')
        return line,
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(trajectory),
                        interval=interval, blit=True)
    anim.save("animation.gif", writer="pillow", fps=20) 
    # Return HTML object for Jupyter display
    return HTML(anim.to_jshtml())

def visualize_trajectory(trajectory, save_int = 100, interval=50):
    pts_history = trajectory[1]
    visualize_evolution(pts_history, save_int = save_int, interval=interval)


import open3d as o3d
import time

def visualize_trajectory_o3d(trajectory,  *, skip = 1, delay = 0.05):
    pts_history = np.asarray(trajectory[1])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_history[0])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    for i in range(0, pts_history.shape[0], skip):
        new_points = pts_history[i]
        # Update points (simulate optimization)
        
        pcd.points = o3d.utility.Vector3dVector(new_points)

        # Update the visualizer
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        # Slow down for visibility
        time.sleep(delay)

    # Close the visualizer
    vis.destroy_window


def visualize_alpha_o3d(points, delay = 0.05):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    alpha = 0.1
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)

    # Add the initial geometry to the visualizer
    vis.add_geometry(mesh)

    # Simulate evolution by varying alpha
    for alpha in np.linspace(0.05, 1.0, 20):
        # Generate a new alpha shape
        new_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)

        # Update the visualizer
        vis.remove_geometry(mesh)  # Remove the old mesh
        mesh = new_mesh  # Update the reference to the new mesh
        vis.add_geometry(mesh)  # Add the new mesh

        vis.poll_events()
        vis.update_renderer()

        # Pause to simulate evolution
        time.sleep(delay)

    # Close the visualizer
    vis.destroy_window()
    return tetra_mesh, pt_map

 