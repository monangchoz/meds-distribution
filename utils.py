import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def draw_box(ax, start, dim, color='b', alpha=0.3):
    """Draws a 3D rectangular box given start position and dimensions."""
    x, y, z = start
    dx, dy, dz = dim
    
    # Define vertices of the box
    vertices = np.array([
        [x, y, z],
        [x + dx, y, z],
        [x + dx, y + dy, z],
        [x, y + dy, z],
        [x, y, z + dz],
        [x + dx, y, z + dz],
        [x + dx, y + dy, z + dz],
        [x, y + dy, z + dz]
    ])
    
    # Define 6 faces using the vertex indices
    faces = [
        [vertices[j] for j in [0, 1, 2, 3]],  # Bottom
        [vertices[j] for j in [4, 5, 6, 7]],  # Top
        [vertices[j] for j in [0, 1, 5, 4]],  # Front
        [vertices[j] for j in [2, 3, 7, 6]],  # Back
        [vertices[j] for j in [1, 2, 6, 5]],  # Right
        [vertices[j] for j in [0, 3, 7, 4]],  # Left
    ]
    
    # Create a 3D polygon collection and add to the axis
    poly3d = Poly3DCollection(faces, alpha=alpha, linewidths=0.5, edgecolors='k')
    poly3d.set_facecolor(color)
    ax.add_collection3d(poly3d)

def visualize_container(container_dim, positions, dims):
    """Visualizes the container with boxes inside."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Draw container boundary
    draw_box(ax, [0, 0, 0], container_dim, color='gray', alpha=0.1)

    # Assign different colors to each box
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'orange', 'purple']
    
    for i, (start, dim) in enumerate(zip(positions, dims)):
        color = colors[i % len(colors)]  # Cycle through colors
        draw_box(ax, start, dim, color=color, alpha=0.5)

    # Set limits
    ax.set_xlim([0, container_dim[0]])
    ax.set_ylim([0, container_dim[1]])
    ax.set_zlim([0, container_dim[2]])

    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Container Visualization')

    plt.show()
