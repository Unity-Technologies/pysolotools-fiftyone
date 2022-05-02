"""Pytest test configuration script."""
import os
from typing import List

import pytest
from _pytest.config import Config, argparsing


def pytest_addoption(parser: argparsing.Parser) -> None:
    """Adds command line options to pytest.

    Args:
        parser (argparsing.Parser): The pytest parser.
    """
    parser.addoption(
        "--lint-only",
        action="store_true",
        default=False,
        help="Only run linting checks.",
    )
    parser.addoption(
        "--print-env-var",
        action="store_true",
        default=False,
        help="Print the value of the TEMPLATE_VAR environment variable.",
    )


def pytest_configure(config: Config) -> None:
    """Configures pytest execution.

    Args:
        config (Config): The pytest session configuration object.
    """
    if config.option.print_env_var:
        print(f"TEMPLATE_VAR={os.environ['TEMPLATE_VAR']}")


def pytest_collection_modifyitems(
    config: Config,
    items: List[pytest.Item],
) -> None:
    """Modifies the collection of tests to run.

    Args:
        config (Config): The pytest configuration object.
        items (List[Item]): The list of tests to run.
    """
    if config.option.lint_only:
        lint_items = []
        linters = ["bandit", "black", "flake8", "isort", "pydocstyle", "pylint", "mypy"]
        for linter in linters:
            if config.getoption(f"--{linter}"):
                lint_items.extend(
                    [item for item in items if item.get_closest_marker(linter)]
                )
        items[:] = lint_items
