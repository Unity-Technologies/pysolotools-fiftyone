"""Template module."""
import argparse
import logging
import sys
from typing import Dict, List

logger = logging.getLogger(__name__)


class ExampleClass:
    """An example class."""

    def __init__(self, name: str) -> None:
        """Class constructor.

        Args:
            name (str): name of the class.
        """
        self.name = name

    def get_name(self) -> str:
        """Returns the name of the class.

        Returns:
            str: Name of the class.
        """
        return self.name

    @staticmethod
    def get_list(length: int) -> List[int]:
        """Returns a list of integers.

        Args:
            length (int): Length of the list.

        Returns:
            List[int]: List ranging from 0 to length.
        """
        return list(range(length + 1))

    @staticmethod
    def get_dict(length: int) -> Dict[str, int]:
        """Returns a dictionary of integers.

        Args:
            length (int): Length of the dictionary.

        Returns:
            Dict[str, int]: Dictionary with keys and values ranging from 0 to length.
        """
        return {str(i): i for i in range(length + 1)}

    @staticmethod
    def get_list_cli() -> None:
        """Prints a list of integers ranging from 0 to length.

        Uses a command line argument parser to fetch the length parameter.
        """
        arg_parser = argparse.ArgumentParser(
            description="Prints a list of integers ranging from 0 to length."
        )
        arg_parser.add_argument(
            "--length",
            "-l",
            type=int,
            help="Length of the list (default: %(default)s).",
            required=True,
        )
        args = arg_parser.parse_args(sys.argv[1:])
        logger.info(" ".join(str(i) for i in range(args.length + 1)))
