import json
import os


def get_cell_points3D(cellParamsJSON):
    curr_dir = os.getcwd()
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    os.chdir(dir_path)
    with open(f'cells/{cellParamsJSON}', 'r') as f:
        cellJSON = json.load(f)
    os.chdir(curr_dir)

    pts = []
    for name, sec in cellJSON['secs'].items():
        for pt in sec['geom']['pt3d']:
            pts.append(pt[:3])

    return pts