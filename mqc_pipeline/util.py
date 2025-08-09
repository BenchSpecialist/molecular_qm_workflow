import os
import sys
import time
import logging
import tracemalloc
from loguru import logger as _loguru_logger
from functools import wraps
from contextlib import contextmanager
from typing import Callable, TypeVar, Any, Tuple

T = TypeVar('T')

# Remove the default loguru handler
_loguru_logger.remove()

# Global variable to track if logger is configured
_logger_configured = False


def get_default_logger(log_file="mqc_pipeline.log") -> Any:
    global _logger_configured

    if not _logger_configured and log_file:
        # Add a new handler that logs to a file, the saves generic information
        _loguru_logger.add(
            log_file,
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}",
            backtrace=False,  # Disable traceback unless needed
            diagnose=False,  # Disable deep inspection of tracebacks
        )
        _logger_configured = True

    return _loguru_logger


def setup_logger(name: str,
                 log_file: str = None,
                 level: int = logging.DEBUG,
                 stream: bool = False,
                 format: str = '%(message)s',
                 datefmt: str = None):
    """
    Setup a standard logging.Logger instance.

    :param name: Logger name
    :param log_file: Optional log file path
    :param level: Logging level
    :param stream: Whether to add stream handler
    :return: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(fmt=format, datefmt=datefmt)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if stream:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    return logger


def timeit(func: Callable[..., T], *args: Any,
           **kwargs: Any) -> Tuple[T, float]:
    """
    Execute a function and return both its result and execution time in seconds
    rounding to 4 decimal places.

    :param func: Function to execute
    :param args: Positional arguments for the function
    :param kwargs: Keyword arguments for the function

    :return: Tuple of (function_result, execution_time_in_seconds)
    """
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    duration = time.perf_counter() - start_time
    return result, round(duration, 4)


def get_human_readable_size(size_bytes) -> str:
    """
    Convert file size to human-readable format.

    :param path: Path to the file
    :return: Human-readable size string, e.g., "1.23 MB"
    """
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    size = float(size_bytes)
    unit_index = 0

    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1

    return f"{size:.2f} {units[unit_index]}"


def profiler(func):
    """
    Decorator to profile a function's execution time and memory usage.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        start_time = time.perf_counter()

        try:
            result = func(*args, **kwargs)
        finally:
            current_mem, max_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            end_time = time.perf_counter()
            execution_time = end_time - start_time

            print(f"  [Profiling] Function: {func.__name__}")
            print(
                f"  [Profiling] Execution Time: {execution_time:.4f} seconds")
            print(
                f"  [Profiling] Peak Memory Usage: {get_human_readable_size(max_bytes)}"
            )

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


def get_optimal_workers(total_inputs: int, min_items_per_worker: int,
                        cpu_count: int) -> int:
    """
    Calculate optimal number of worker processes based on workload and CPU count.

    :param total_inputs: Total number of inputs to process
    :param min_items_per_worker: Minimum inputs per worker for efficient processing
    :param cpu_count: Number of CPU cores available

    :return: Recommended number of worker processes
    """
    # Calculate workers needed based on minimum batch size
    workers_by_batch = max(1, total_inputs // min_items_per_worker)

    # Limit by available CPUs (N + 4 rule)
    workers_by_cpu = cpu_count + 4

    # Take the minimum of both constraints
    return min(workers_by_batch, workers_by_cpu)


def write_df_to_parq_duckdb(df, output_path) -> None:
    """
    Write a DataFrame (Polars or pandas) to a Parquet file using DuckDB.

    This function efficiently converts the input DataFrame to Arrow format
    and uses DuckDB for optimized Parquet file writing with standard-compliant
    metadata.
    """
    import duckdb
    from pathlib import Path

    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    temp_view_name = "temp_df"

    # Convert DataFrame to PyArrow table based on its type
    if hasattr(df, 'to_arrow'):
        # Polars DataFrame
        arrow_table = df.to_arrow()
    elif hasattr(df, 'to_arrow_table'):
        # pandas DataFrame with pyarrow installed
        arrow_table = df.to_arrow_table()
    elif hasattr(df, 'to_dict'):
        import pyarrow as pa
        # pandas DataFrame without pyarrow - slower fallback
        arrow_table = pa.Table.from_pandas(df)
    else:
        raise TypeError(
            f"Unsupported DataFrame type: {type(df)}. Must be a Polars DataFrame or pandas DataFrame."
        )

    # Register the Arrow table as a DuckDB view
    duckdb.register(temp_view_name, arrow_table)

    try:
        # Save as Parquet (with standard-compliant metadata)
        duckdb.sql(
            f"COPY {temp_view_name} TO '{output_path}' (FORMAT PARQUET);")
    finally:
        # Clean up the registration
        duckdb.sql(f"DROP VIEW IF EXISTS {temp_view_name}")
