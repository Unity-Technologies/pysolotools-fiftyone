"""Python template project."""
import importlib.metadata
import logging

from . import module  # noqa: F401

logging.basicConfig(format="%(message)s", level=logging.DEBUG)

__version__ = importlib.metadata.version("python-template")
