from sl_utils.version import get_git_version
import os
import importlib
import pkgutil

MODULE_DIR = os.path.dirname(__file__)

__version__ = get_git_version(MODULE_DIR)
__author__ = "Paul Golder"
__author_github__ = "https://github.com/Hysnap"
__description__ = "Visualisations for the project."

__all__ = []
for _, module_name, _ in pkgutil.iter_modules(__path__):
    module = importlib.import_module(f"{__name__}.{module_name}")
    globals()[module_name] = module
    __all__.append(module_name)