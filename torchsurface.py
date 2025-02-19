import torch


from pykeops.torch import LazyTensor



def repulsion_potential_k(x, epsilon=1e-6):
  N, D = x.shape
  # Create LazyTensors
  x_i = LazyTensor(x.view(N, 1, D))
  x_j = LazyTensor(x.view(1, N, D))
  # q_i = LazyTensor(q.view(N, 1, 1))
  #q_j = LazyTensor(q.view(1, N, 1))

  D_ij = ((x_i - x_j)**2).sum(dim=2)
  mask = (D_ij != 0)
  # Compute distances and potential in one step
  #R_ij = ((x_i - x_j)**2).sum(dim=2).sqrt()
  V_ij = 1.0/ (D_ij.sqrt() + epsilon) * mask
  #V_ij = 1.0/ (R_ij + epsilon)

  # Compute total potential
  V_total = V_ij.sum(dim=1).sum()
  return V_total/N/(N-1)
# Define the surface function (e.g., unit sphere: x^2 + y^2 + z^2 - 1 = 0)
def surface_function(points):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    return x**2 + y**2 + z**2 - 1  # Example: Sphere

# Compute surface loss
def surface_loss(points):
    f_values = surface_function(points)
    return (f_values**2).mean()  # Mean squared error for surface consistency

# Compute Coulomb-like repulsion potential
def repulsion_potential(points, epsilon=1e-6):
    # Pairwise distances
    distances = torch.cdist(points, points, p=2)  # Euclidean distance matrix
    distances = distances**2 + epsilon  # Avoid division by zero
    inv_distances = 1.0 / distances  # Coulomb potential
    repulsion = inv_distances.sum() - inv_distances.diag().sum()  # Exclude self-interactions
    n = points.size(0)
    return repulsion / (n * (n - 1))  # Normalize by number of pairs

# Initialize random points in 3D space
n_points = 100
points = torch.rand((n_points, 3), requires_grad=True) * 2 - 1  # Random points in [-1, 1]^3

# Optimizer
optimizer = torch.optim.Adam([points], lr=0.01)

# Training loop
lambda_reg = 0.1  # Weight for repulsion term
for epoch in range(1000):
    optimizer.zero_grad()

    # Compute losses
    surface_loss_value = surface_loss(points)
    repulsion_value = repulsion_potential_k(points)
    total_loss = surface_loss_value + lambda_reg * repulsion_value

    # Backpropagation and optimization
    total_loss.backward()
    optimizer.step()

    # Log progress
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Total Loss: {total_loss.item()}, Surface Loss: {surface_loss_value.item()}, Repulsion: {repulsion_value.item()}")

# Final points
final_points = points.detach().numpy()