'''
This module contains utilities for dispalying calculation progress

'''

import sys
import time

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