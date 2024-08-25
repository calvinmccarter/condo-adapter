# -*- coding: utf-8 -*-
import os
from setuptools import setup, find_packages

def readme():
    with open("README.md", encoding="utf-8") as readme_file:
        return readme_file.read()

configuration = {
    "name": "condo",
    "version": "0.8.0",
    "description": "Confounded domain adaptation",
    "long_description": readme(),
    "long_description_content_type": "text/markdown",
    "classifiers": [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved",
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    "keywords": "confounding, domain adaptation, batch correction",
    "url": "http://github.com/calvinmccarter/condo-adapter",
    "author": "Calvin McCarter",
    "author_email": "mccarter.calvin@gmail.com",
    "maintainer": "Calvin McCarter",
    "maintainer_email": "mccarter.calvin@gmail.com",
    "packages": ["condo"],
    "install_requires": [
        "miceforest<6.0.0",
        "numpy",
        "pandas",
        "pytorch-minimize>=0.0.2",
        "scipy",
        "scikit-learn",
        "torch>=1.4.0",
    ],
    "license": "by-nc-sa-4.0",
    "ext_modules": [],
    "cmdclass": {},
    "test_suite": "nose.collector",
    "tests_require": ["nose"],
    "data_files": (),
    "zip_safe": True,
    "python_requires" : ">=3.8",
}

setup(**configuration)
