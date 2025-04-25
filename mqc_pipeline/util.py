import time
import subprocess
from functools import wraps


def gpu_available():
    try:
        output = subprocess.check_output("nvidia-smi", shell=True).decode()
        print("GPU is available:\n", output)
        return True
    except subprocess.CalledProcessError:
        print("No GPU available or NVIDIA drivers not installed")
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
            print(f"{func.__name__}: {execution_time:.4f} seconds")

        # Return the original function's result
        return result

    return wrapper
