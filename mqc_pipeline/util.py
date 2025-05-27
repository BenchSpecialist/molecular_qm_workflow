import os
import sys
import time
import logging
import subprocess
from loguru import logger
from functools import wraps
from contextlib import contextmanager
from typing import Callable, TypeVar, Any, Tuple

T = TypeVar('T')
# Remove the default loguru handler
logger.remove()

# Add a new handler that logs to a file, the saves generic information
logger.add(
    "mqc_pipeline.log",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}",
    backtrace=False,  # Disable traceback unless needed
    diagnose=False  # Disable deep inspection of tracebacks
)


def setup_logger(name, log_file=None, level=logging.DEBUG, stream=False):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter('%(message)s')

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if stream:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    return logger


def has_nvidia_gpu():
    try:
        # Check if 'nvidia-smi' command is available and outputs something
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        return result.returncode == 0 and len(result.stdout) > 0
    except FileNotFoundError:
        return False


def timeit(func: Callable[..., T], *args: Any,
           **kwargs: Any) -> Tuple[T, float]:
    """
    Execute a function and return both its result and execution time.

    :param func: Function to execute
    :param args: Positional arguments for the function
    :param kwargs: Keyword arguments for the function
    :return: Tuple of (function_result, execution_time_in_seconds)
    """
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    duration = time.perf_counter() - start_time
    return result, duration


def timer(func):
    """
    Decorator to profile execution time of a function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()

        try:
            result = func(*args, **kwargs)
        finally:
            end_time = time.perf_counter()
            execution_time = end_time - start_time

            # Print profiling information
            print(
                f"\nProfiled `{func.__name__}`: {execution_time:.4f} seconds")

        # Return the original function's result
        return result

    return wrapper


@contextmanager
def change_dir(new_dir: str):
    """
    Context manager to change the working directory.
    """
    new_dir = os.path.abspath(new_dir)
    old_dir = os.getcwd()
    os.makedirs(new_dir, exist_ok=True)
    try:
        os.chdir(new_dir)
        yield
    finally:
        os.chdir(old_dir)
