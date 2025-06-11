import os
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path

sim_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(sim_path)

def set_cwd_to_root_for_func(func, *args, **kwargs):
    curr_dir = os.getcwd()
    os.chdir(root_path)
    retval = func(*args, **kwargs)
    os.chdir(curr_dir)
    return retval

def set_cwd_to_sim_for_func(func, *args, **kwargs):
    curr_dir = os.getcwd()
    os.chdir(sim_path)
    retval = func(*args, **kwargs)
    os.chdir(curr_dir)
    return retval

@contextmanager
def set_cwd(directory):
    original_cwd = os.getcwd()
    if directory == "sim":
        path = sim_path
    elif directory == "root":
        path = root_path
    elif type(directory) == str:
        path = os.path.abspath(directory)
    else:
        path = original_cwd
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original_cwd)

@contextmanager
def suppress_stdout():
    """A context manager that redirects stdout to devnull"""
    with open(os.devnull, 'w') as fnull:
        with redirect_stdout(fnull) as out:
            yield out

@contextmanager
def suppress_stderr():
    """A context manager that redirects stderr to devnull"""
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err:
            yield err

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)