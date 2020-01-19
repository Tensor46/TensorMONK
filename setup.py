#!/usr/bin/env python
import os
import io
import re
import sys
from setuptools import setup, find_packages


with open('README.md', 'r') as fh:
    long_description = fh.read()

requirements = [
    'pytorch>= 0.4.1',
    'torchvision',
    'visdom',
]

setup(
    # Metadata
    name='tensormon',
    version='0.1',
    author='Vikas Gottemukkula',
    author_email='tensor46@gmail.com',
    url='https://github.com/Tensor46/TensorMONK',
    description='TenorMonk is a collection of deep learning architectures',
    long_description=long_description,
    license='BSD',

    # Package info
    packages=find_packages(exclude=('test',)),

    zip_safe=True,
    install_requires=requirements,
    extras_require={},
)