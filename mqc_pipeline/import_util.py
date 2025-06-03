from types import ModuleType
from functools import lru_cache

from .util import get_default_logger

logger = get_default_logger()


@lru_cache(maxsize=1)
def import_rks_uks() -> tuple[ModuleType, ModuleType]:
    """
    Import and return RKS and UKS modules from PySCF or GPU4PySCF.

    This function attempts to import GPU-accelerated modules first, falling back
    to standard PySCF modules if GPU4PySCF is not available. The imports are
    cached using lru_cache to avoid repeated slow imports (GPU4PySCF modules take
    ~5.4 seconds to import on Fluidstack compute nodes) and dynamic imports are
    used to optimize startup time.
    """
    try:
        from gpu4pyscf.dft import rks, uks
        logger.info("Using GPU-accelerated PySCF.")
        return rks, uks
    except (ImportError, AttributeError):
        from pyscf.dft import rks, uks
        return rks, uks


@lru_cache(maxsize=1)
def import_cupy() -> tuple[ModuleType | None, bool]:
    """
    Import and return CuPy module with availability status.

    This function attempts to import CuPy and verify GPU device availability.
    The import is cached using lru_cache to avoid repeated slow imports and
    device checks. Returns both the module (if available) and a boolean flag
    indicating GPU availability status.
    """
    try:
        import cupy
        _ = cupy.cuda.runtime.getDeviceCount()
        return cupy, True
    except (ImportError, AttributeError):
        # cupy not installed or not configured correctly
        logger.warning(
            "CuPy is not available. GPU-accelerated calculations will not be used."
        )
        return None, False
