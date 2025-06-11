import numpy as np
from math import ceil, sin, radians

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