from os import path
from autoconf import conf

import pytest

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance = conf.Config(
        path.join(directory, "config"), path.join(directory, "output")
    )
