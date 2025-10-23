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
    import sys

    # detect whether installation is running
    cond1 = "PIP_BUILD_TRACKER" in os.environ  # triggered by pip
    cond2 = os.path.join("uv", "builds-v") in sys.executable
    cond3 = "_PYPROJECT_HOOKS_BUILD_BACKEND" in os.environ  # triggered by uv pip install

    if  any((cond1, cond2, cond3)):
        pass
    else:
        # raise the original exception
        raise
