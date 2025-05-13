import time
import subprocess
from functools import wraps
from loguru import logger

# Remove the default loguru handler
logger.remove()

# Add a new handler that logs to a file with your desired format
logger.add(
    "mqc_pipeline.log",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}",
    backtrace=False,  # Disable traceback unless needed
    diagnose=False  # Disable deep inspection of tracebacks
)


def has_nvidia_gpu():
    try:
        # Check if 'nvidia-smi' command is available and outputs something
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        return result.returncode == 0 and len(result.stdout) > 0
    except FileNotFoundError:
        return False


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
