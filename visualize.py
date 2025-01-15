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
    
    # Return HTML object for Jupyter display
    return HTML(anim.to_jshtml())

def visualize_trajectory(trajectory, save_int = 100, interval=50):
    pts_history = trajectory[1]
    visualize_evolution(pts_history, save_int = save_int, interval=interval)
