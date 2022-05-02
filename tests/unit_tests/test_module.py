"""Unit tests for the template module."""
import logging
from unittest import mock

import pytest

import template


@pytest.mark.unit_test
class TestModule:
    """Class testing the template main module."""

    @staticmethod
    def test_get_name() -> None:
        """Tests the get_name method."""
        example = template.module.ExampleClass("test")
        assert example.get_name() == "test"

    @staticmethod
    def test_get_list() -> None:
        """Tests the get_list method."""
        example = template.module.ExampleClass("test")
        expected = list(range(11))
        assert example.get_list(10) == expected

    @staticmethod
    def test_get_dict() -> None:
        """Tests the get_dict method."""
        example = template.module.ExampleClass("test")
        expected = {str(i): i for i in range(11)}
        assert example.get_dict(10) == expected

    @staticmethod
    @mock.patch("sys.argv", ["", "-l", "10"])
    def test_get_list_cli(caplog: pytest.LogCaptureFixture) -> None:
        """Tests the get_list_cli method."""
        example = template.module.ExampleClass("test")
        with caplog.at_level(logging.DEBUG):
            example.get_list_cli()
            assert caplog.messages == ["0 1 2 3 4 5 6 7 8 9 10"]

    def long_line_function(self) -> None:
        """This function is supposed to test that 119 characters are accepted by the template. There is not much more..

        Returns:
            None
        """
