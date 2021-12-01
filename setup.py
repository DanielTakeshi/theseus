#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import setuptools


def parse_requirements_file(path):
    with open(path) as f:
        reqs = []
        for line in f:
            line = line.strip()
            reqs.append(line.split("==")[0])
    return reqs


reqs_main = parse_requirements_file("requirements/main.txt")
reqs_dev = parse_requirements_file("requirements/dev.txt")
root_dir = Path(__file__).parent
version = (root_dir / "version.txt").read_text().strip()

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="theseus",
    version=version,
    author="Meta Research",
    description="A library for differentiable nonlinear optimization.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/facebookresearch/theseus",
    keywords="differentiable optimization, nonlinear least squares, factor graphs",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=reqs_main,
    extras_require={"dev": reqs_main + reqs_dev},
)