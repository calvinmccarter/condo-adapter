# -*- coding: utf-8 -*-
import os
from setuptools import setup, find_packages

try:
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
except IOError:
    long_description = ""

# Extract version. Cannot import directly because of import error.
root_dir = os.path.dirname(__file__)
with open(os.path.join(root_dir, "condo/__init__.py"), "r") as f:
    for line in f.readlines():
        if line.startswith("__version__"):
            version = line.split("=")[-1].strip().strip('"')
            break

PACKAGES = find_packages(exclude=("tests.*", "tests"))

setup(
    name="condo",
    version=version,
    description="Library to perform confounded domain adaptation.",
    license="apache-2.0",
    author="Calvin McCarter",
    packages=PACKAGES,
    install_requires=["torch>=1.4.0"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/calvinmccarter/condo-adapter",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)