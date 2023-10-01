#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import io
from codecs import open
from setuptools import setup, find_packages

REQUIRES = [
    'numpy>=0.15.0',
    'scikit-learn>=0.19.0',
    'pandas>=0.22.0',
    'networkx>=2.2',
    'geopy>=1.17.0',
    'geopandas>=0.3.0'
]

PACKAGE_DATA = {
    '': [
        'LICENSE',
    ],
}

here = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = '\n' + f.read()

about = {}
with open(os.path.join(here, 'sqterritory', '__version__.py'), 'r', 'utf-8') as f:
        exec(f.read(), about)

setup(
    name=about['__title__'],
    version=about['__version__'],
    description=about['__description__'],
    long_description=long_description,
    url=about['__url__'],
    author=about['__author__'],
    author_email=about['__email__'],
    license=about['__license__'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ],
    keywords='min cost flow demo',
    packages=find_packages(),
    install_requires=REQUIRES,
    python_requires=">=3.6",
    include_package_data=True,
    package_dir={'sqterritory': 'sqterritory'},
    package_data=PACKAGE_DATA,
)
