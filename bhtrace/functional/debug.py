'''
Debug decorator. 
'''
import logging
import functools
import inspect

# Configure logger
logger = logging.getLogger('debug_decorator')
logger.setLevel(logging.DEBUG)

# Create a file handler which logs even debug messages
# In case the file is not created, check permissions
fh = logging.FileHandler('debug_log.txt')
fh.setLevel(logging.DEBUG)

# Create formatter and add it to the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)

# Add the handler to the logger
if not logger.handlers:
    logger.addHandler(fh)

def debug(func):
    """
    A decorator that logs the function's call signature and return value.
    """
    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        logger.debug(f"--- Calling {func.__name__} ---")
        logger.debug("Arguments:")
        for name, value in bound_args.arguments.items():
            logger.debug(f"  {name}:")
            logger.debug(f"    {repr(value)}")

        value = func(*args, **kwargs)

        logger.debug(f"--- {func.__name__} returned ---")
        if isinstance(value, tuple):
            logger.debug("Outputs (unpacked tuple):")
            if not value:
                logger.debug("  (empty tuple)")
            for i, item in enumerate(value):
                logger.debug(f"  output_{i}:")
                logger.debug(f"    {repr(item)}")
        else:
            logger.debug("Output:")
            logger.debug(f"  {repr(value)}")
        logger.debug(f"--- End of {func.__name__} call ---")

        return value
    return wrapper_debug
