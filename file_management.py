import os

def reach_out_for_func(func, *args, **kwargs):
    curr_dir = os.getcwd()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir_path)
    retval = func(*args, **kwargs)
    os.chdir(curr_dir)
    return retval