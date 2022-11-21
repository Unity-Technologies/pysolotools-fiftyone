#!/usr/bin/env python
"""
Installs pysolotools-fiftyone
"""

import io
import os
from os.path import dirname, realpath

from setuptools import find_packages, setup

# Package meta-data.
NAME = "pysolotools-fiftyone"
DESCRIPTION = "Voxel fiftyone integration for SOLO"
URL = "https://https://github.com/Unity-Technologies/pysolotools-fiftyone"
EMAIL = "computer-vision@unity3d.com"
AUTHOR = "Unity Technologies"
REQUIRES_PYTHON = ">=3.8"
VERSION = "0.3.22"


here = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = "\n" + f.read()


def _read_requirements():
    requirements = f"{dirname(realpath(__file__))}/requirements.txt"
    with open(requirements) as f:
        results = []
        for line in f:
            line = line.strip()
            if "-i" not in line:
                results.append(line)
        return results


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    py_modules=[NAME],
    include_package_data=True,
    license="MIT",
    install_requires=_read_requirements(),
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    packages=find_packages(),
    entry_points={
        "console_scripts": ["pysolotools-fiftyone=pysolotools_fiftyone.solo_fiftyone:cli"]
    },
)
