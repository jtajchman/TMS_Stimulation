from scipy.spatial.transform import Rotation as R
import warnings
import threading

def rotate_coords_from_axis(axis, coords):
    # Cartesian coordinate transformation that rotates the z axis to the defined axis
    # Azimuthal orientation about this new axis is different from taking a direct rotation, but this does not matter here
    r = get_rotation(axis)
    return rotate_coords(r, coords)

def get_rotation(axis):
    # with threading.RLock():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r, _ = R.align_vectors([axis], [[0, 0, 1]])
    return r

def rotate_coords(r, coords):
    return r.apply(coords)