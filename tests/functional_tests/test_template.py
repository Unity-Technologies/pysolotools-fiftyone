"""Functional tests for the template module."""
import logging
from unittest import mock

import pytest

import template


@pytest.mark.functional_test
@mock.patch("sys.argv", ["", "-l", "10"])
def test_template(caplog):
    """A dummy functional test."""
    obj1 = template.module.ExampleClass("object1")
    obj2 = template.module.ExampleClass("object2")

    assert obj1.get_name() != obj2.get_name()

    with caplog.at_level(logging.DEBUG):
        obj1.get_list_cli()
        obj2.get_list_cli()
        assert caplog.messages == ["0 1 2 3 4 5 6 7 8 9 10"] * 2

    dict1 = obj1.get_dict(10)
    list2 = obj2.get_list(10)

    assert list(dict1.values()) == list2
