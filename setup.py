# -*- coding: utf-8 -*-
"""Setup file."""
from pathlib import Path

from setuptools import find_packages, setup

setup(
    name="torchic",
    version="0.1.0",
    description="Small logistic learning framework regression in torch",
    author="Stéphan Tulkens",
    author_email="stephantul@gmail.com",
    url="https://github.com/stephantul/torchic",
    license="MIT",
    packages=find_packages(include=["torchic"]),
    project_urls={
        "Source Code": "https://github.com/stephantul/torchic",
        "Issue Tracker": "https://github.com/stephantul/torchic/issues",
    },
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=["torch"],
    zip_safe=True,
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
)
