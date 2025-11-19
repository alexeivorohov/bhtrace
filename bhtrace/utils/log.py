import logging
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn

class Logger:
    def __init__(self, log_file="default.log", silent=True, max_messages=256, show_nn_Modules=False):
        
        self.log_file = log_file
        self.silent = silent
        self.max_messages = max_messages
        self.show_nn_Modules = show_nn_Modules

        self.active = False
        self._calls_ = {}
        self.n_msg = 0

        # Configure logging
        self.logger = logging.getLogger(log_file)
        self.logger.propagate = False # prevent passing to root logger
        self.logger.setLevel(logging.DEBUG)
        
        if not self.logger.handlers:
            # Use a unique name for the logger to avoid conflicts.
            # Using the log_file path is a good way to ensure uniqueness.
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    @contextmanager
    def log(self):
        self.clear_log()
        self.zero()
        self.on()
        try:
            yield
        finally:
            self.off()

    def on(self):
        self.active = True

    def off(self):
        self.active = False
        for handler in self.logger.handlers:
            handler.flush()

    def set_silent(self, silent: bool = True):
        self.silent = silent

    def zero(self):
        self._calls_ = {}
        self.n_msg = 0

    def check_quota(self):
        self.n_msg += 1 
        if self.n_msg >= self.max_messages:
            self.off() 
            msg = 'Logger acheived message limit'
            if not self.silent:
                print(msg)
            self.logger.debug(msg)

    def clear_log(self):
        # Overwrite the file to clear it
        with open(self.log_file, 'w') as f:
            f.write("")

    def trace(self, name):
        def log_fn(func):
            def wrapper(*args, **kwargs):
                if self.active:
                    if name not in self._calls_:
                        self._calls_[name] = 0

                    # Log function call details
                    msg = f'>>> {name} call {self._calls_[name]}: \n Args: \n'
                    msg += self.format_args(args, kwargs)

                    if not self.silent:
                        print(msg)
                    self.logger.debug(msg)

                result = func(*args, **kwargs)

                if self.active:
                    # Log the result of the function call
                    result_msg = f'Result: \n'
                    result_msg += self.format_result(result)

                    if not self.silent:
                        print(result_msg)
                    self.logger.debug(result_msg)
                    self._calls_[name] += 1
                
                self.check_quota()
                return result
            return wrapper

        return log_fn

    def format_args(self, args, kwargs):
        msg = ""
        for i, arg in enumerate(args):
            msg += f'\t arg[{i}]: ' + self.format_value(arg, prefix = ' ') + '\n'
        for k, arg in kwargs.items():
            msg += f'\t kwarg[{k}] :' + self.format_value(arg, prefix = ' ') + '\n'
        return msg
    
    def format_value(self, value, prefix='\t'):
        msg = prefix
        match value:

            case nn.Module():
                msg += f'(nn.Module): {value.__class__.__name__}'
                if self.show_nn_Modules:
                    msg += f'\n {value.__repr__()}'
            
            case torch.Tensor():
                msg += f'(torch.Tensor): ' + self.format_tensor(value.detach())
            
            case np.ndarray():
                msg += f'(np.ndarray): ' + self.format_array(value)

            case list():
                msg += '(List): ['
                for i, v in enumerate(value):
                    msg += self.format_value(v, prefix='\t')
                msg += ']'

            case tuple():
                msg += '(Tuple): ('
                for i, v in enumerate(value):
                    msg += f'{i}: ' + self.format_value(v, prefix='\t')
                msg += ')'

            case dict():
                msg += '(Dict): {'
                for k, v in value.items():
                    msg += f'{k}: ' + self.format_value(v, prefix='\t')
                msg += '}'

            case _:
                msg += f'(Other): {repr(value)}'

        return msg

    def format_result(self, result):
        
        return self.format_value(result, prefix='\t output:')

    def format_tensor(self, tensor: torch.Tensor):
        msg = f'shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}, '
        
        msg += f'requires_grad={tensor.requires_grad}, gradfn={tensor.grad_fn}, '

        msg += f'\n data = \n {tensor.data} \n grad = \n {tensor.grad}'

        return msg
    
    def format_array(self, array: np.ndarray):
        msg = f'shape={array.shape}, dtype={array.dtype}, '
        
        msg += f'ndim={array.ndim}, size={array.size}, '
        
        msg += f'\n data = \n{array}'

        return msg

LOG = Logger()
'''
Default global logger

Basic usage:

1. Add function in logger with name 'NAME':

    @LOG.trace('NAME')
    def function_to_trace(...): ....

2. Complete code within LOG.log() context manager:

    with LOG.log():
        function_to_trace(a, 10, 'str')

3. See inputs and outputs in default.log file

'''