from scipy.spatial.transform import Rotation as R
from .spherical_cartesian_conversion import normalize
import numpy as np
import warnings
import threading

def rotate_coords_from_axis_to_pos_z(axis, coords):
    # Cartesian coordinate transformation that rotates the coords aligned to the defined axis to the positive z axis
    r = get_rotation_to_pos_z(axis)
    return rotate_coords(r, coords)

def rotate_coords_from_pos_z_to_axis(axis, coords):
    # Cartesian coordinate transformation that rotates the coords aligned to the positive z axis to the defined axis
    r = get_inverted_rotation(get_rotation_to_pos_z(axis))
    return rotate_coords(r, coords)

def get_rotation_to_pos_z(somatodendritic_axis: list, azimuthal_axis: list = [1, 0, 0]):
    assert somatodendritic_axis != azimuthal_axis
    new_z = list(normalize(*somatodendritic_axis))
    new_y = list(normalize(*np.cross(new_z, azimuthal_axis)))
    new_x = list(np.cross(new_y, new_z))

    with threading.RLock():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = R.from_matrix([new_x, new_y, new_z])
    return r

def get_inverted_rotation(r: R):
    return R.from_matrix(np.linalg.inv(r.as_matrix()))

def get_rotation_old(somatodendritic_axis: list):
    with threading.RLock():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r, _ = R.align_vectors([somatodendritic_axis], [[0, 0, 1]])
    return r

def rotate_coords(r: R, coords):
    return r.apply(coords)