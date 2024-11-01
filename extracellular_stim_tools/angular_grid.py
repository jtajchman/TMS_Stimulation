import numpy as np
from numpy import pi
from math import ceil, sin, radians, degrees
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt 
from .spherical_cartesian_conversion import norm_spherical_to_cartesian

def get_angles(polar_resolution, azimuthal_resolution):
    num_polar_angles = ceil(180/polar_resolution)+1
    polar_angles = np.linspace(0, 180, num_polar_angles)
    angles = []
    for polar_angle in polar_angles:
        num_azimuthal_angles = get_num_azimuthal_angles(polar_angle, azimuthal_resolution)
        azimuthal_angles = np.linspace(0, 360, num_azimuthal_angles, endpoint=False).tolist()
        angles.append([polar_angle, azimuthal_angles])
    return angles

def get_num_azimuthal_angles(polar_angle, azimuthal_resolution):
    num_azimuthal_angles = ceil(round(360*sin(radians(polar_angle))/azimuthal_resolution, 9))
    if num_azimuthal_angles == 0: # Special case for polar_angle == 0 or 180
        return 1
    else:
        return num_azimuthal_angles

def unpack_angles(angles):
    return [[angle[0], azimuth] for angle in angles for azimuth in angle[1]]

def main():
    # print(get_num_azimuthal_angles(90, 60))
    angles = get_angles(10, 90)
    angle_list = unpack_angles(angles)
    xs = []
    ys = []
    zs = []
    for angle in angle_list:
        x, y, z = norm_spherical_to_cartesian(angle[0], angle[1])
        xs.append(x)
        ys.append(y)
        zs.append(z)
    
    fig = plt.figure() 
    ax = Axes3D(fig)

    ax.scatter(xs, ys, zs, color='green') 
    ax.set_title("3D plot") 
    ax.set_xlabel('x-axis') 
    ax.set_ylabel('y-axis') 
    ax.set_zlabel('z-axis') 
    plt.show()

if __name__ == '__main__':
    main()