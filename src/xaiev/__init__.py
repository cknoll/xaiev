from .release import __version__

# the package is imported during installation
# however installation happens in an isolated build environment
# where no dependencies are installed.

# this means: no importing the following modules will fail
# during installation. This is OK, but only during installation

try:
    from . import core
    from .core import *
except ImportError:
    import os

    # detect whether installation is running
    # TODO: This does not work with `uv pip install -e .`

    if "PIP_BUILD_TRACKER" in os.environ or "_PYPROJECT_HOOKS_BUILD_BACKEND" in os.environ:
        pass
    else:
        # raise the original exception
        raise
