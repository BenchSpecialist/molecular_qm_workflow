import platform
from typing import Callable

import pytest


def requires_openbabel(reason: str | None = None) -> Callable:
    """
    Decorator to skip tests when OpenBabel pybel import fails.
    The GitHub unittests workflow cannot install openbabel correctly with pyproject.toml,
    so this decorator is used to skip tests when the package is not available.
    """
    try:
        from openbabel import pybel
        skip_condition = False
    except ImportError:
        skip_condition = True

    return pytest.mark.skipif(skip_condition,
                              reason=reason or "OpenBabel not available")


# Platform detection constants
IS_DARWIN: bool = platform.system() == "Darwin"
IS_LINUX: bool = platform.system() == "Linux"
IS_WINDOWS: bool = platform.system() == "Windows"


# Decorator factories with optional custom reasons
def darwin_only(reason: str | None = None) -> Callable:
    """
    Decorator to skip tests on non-Darwin platforms.
    """
    return pytest.mark.skipif(not IS_DARWIN,
                              reason=reason
                              or "Test only runs on macOS/Darwin")


def linux_only(reason: str | None = None) -> Callable:
    """
    Decorator to skip tests on non-Linux platforms.
    """
    return pytest.mark.skipif(not IS_LINUX,
                              reason=reason or "Test only runs on Linux")


def windows_only(reason: str | None = None) -> Callable:
    """
    Decorator to skip tests on non-Windows platforms.
    """
    return pytest.mark.skipif(not IS_WINDOWS,
                              reason=reason or "Test only runs on Windows")
