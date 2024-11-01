from .spherical_cartesian_conversion import (
    cartesian_to_spherical, 
    spherical_to_cartesian, 
    norm_cartesian_to_spherical, 
    norm_spherical_to_cartesian
    )

from scipy.spatial.transform import Rotation as R
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def rotate_coords_from_axis(axis, coords):
    # Cartesian coordinate transformation that rotates the z axis to the defined axis
    r, _ = R.align_vectors([axis], [[0, 0, 1]])
    return r.apply(coords)

def get_rotation(axis):
    r, _ = R.align_vectors([axis], [[0, 0, 1]])
    return r

def rotate_coords(r, coords):
    return r.apply(coords)

def plot_polyh(vertices):
    vertices = np.array(vertices)
    ax = plt.figure().add_subplot(projection="3d")
    ax.set_aspect('equal')
    ax.set_xlim3d([-3, 3])
    ax.set_ylim3d([-3, 3])
    ax.set_zlim3d([-3, 3])
    ax.set_xlabel('x-axis') 
    ax.set_ylabel('y-axis') 
    ax.set_zlabel('z-axis')
    hull = ConvexHull(vertices)
    # draw the polygons of the convex hull
    for s, color in zip(hull.simplices, ['r', 'g', 'b', 'k']):
        tri = Poly3DCollection([vertices[s]])
        tri.set_color(color)
        tri.set_alpha(0.5)
        ax.add_collection3d(tri)
    # draw the vertices
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], marker='o', color='purple')
    plt.show()