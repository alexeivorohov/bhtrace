'''
This module contains utilities for dispalying calculation progress

'''
import os
import sys
import pathlib
import logging

from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from bhtrace.data import Trajectory

def print_status_bar(progress, total, elapsed_time):
    '''
    This function displays and updates status bar.

    Args:
    - progress: int - current progress
    - total: int - total steps
    - elapsed_time: float - elapsed time in seconds
    '''
    bar_length = 20  # Length of the status bar
    filled_length = int(bar_length * progress // total)  # Calculate filled length
    bar = '=' * filled_length + ' ' * (bar_length - filled_length)  # Create the bar
    percentage = (progress / total) * 100  # Calculate percentage
    status = f'\r {percentage:.2f}% [{bar}] | '

    # Estimate remaining time
    if progress == 0:
        info = "N/A tr/s | T: N/A s | ETA N/A s"
    elif progress == total:  
        trs = progress / elapsed_time
        info = f'{trs:.2f} tr/s | T: {elapsed_time:.2f} s | Done !' 
    else:
        trs = progress / elapsed_time
        ETA_t = (total - progress) / trs
        info = f'{trs:.2f} tr/s | T: {elapsed_time:.2f} s | ETA {ETA_t:.2f} s' 

    sys.stdout.write(status + info)
    sys.stdout.flush()

def _env_var_path_override(name: str, is_dir: bool = False, log: logging.Logger = None) -> str | None:

    path = os.environ.get(name)
    if path is None:
        return None


    path = pathlib.Path(path)
    assert path.exists(),\
        f"Detected path override by environment variable {name}, but path does not exist {path}"
    
    if is_dir:
        assert path.is_dir(),\
            f"Path override by environment variable {name} is not a directory: {path}"
    else:
        assert path.is_file(),\
            f"Path override by environment variable {name} is not a file: {path}"

    log = log or logging.getLogger(__file__)

    log.info(
        f"Successfully read path override from environment variable {name}, "
        f"set {name} to {path}"
    )

    return path

def loader(directory) -> Dict[str, 'Trajectory']:
    from bhtrace.data import Trajectory
    try:
        files = os.listdir(directory)
        trajs = {}
        
        file_list = [file for file in files\
                     if os.path.isfile(os.path.join(directory, file))]
        
        print('Read files:\n')
        for file in file_list: 
            print(file)
            traj = Trajectory.load(directory+file)
            trajs[file.replace('.traj', '')] = traj

        return trajs        
    except FileNotFoundError:
        return f"The directory '{directory}' does not exist."
    except PermissionError:
        return f"Permission denied to access '{directory}'."
    except Exception as e:
        print(e)
        return str(e)