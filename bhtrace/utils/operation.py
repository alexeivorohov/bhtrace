'''
This module contains utilities for dispalying calculation progress

'''
import os
import sys
import time
import argparse

from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from bhtrace.trajectory import Trajectory

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

def addparser(docstr):
    '''
    
    '''
    parser = argparse.ArgumentParser(
    description=docstr)
    parser.add_argument('-l', '--load', action='store_true', 
                        help='Indicates if the program should load data from a file (default: False)')
    parser.add_argument('-s', '--save', action='store_true', 
                        help='Indicates if the program should overwrite existing saves (default: True)')
        
    return parser

def loader(directory) -> Dict[str, 'Trajectory']:
    from bhtrace.trajectory import Trajectory
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