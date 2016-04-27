#!/usr/bin/env python
import sys
if sys.version_info < (3, 4):
    raise Exception('exa requires Python 3.4+')
from setuptools import setup, find_packages
from atomic import __version__


try:
    setup(
        name='atomic',
        version=__version__,
        description='Computational chemistry functionality for exa',
        author='Tom Duignan & Alex Marchenko',
        author_email='exa.data.analytics@gmail.com',
        url='https://exa-analytics.github.io/atomic',
        packages=find_packages(),
        package_data={'atomic': ['nbextensions/*.js']},
        include_package_data=True,
        license='Apache License Version 2.0'
    )
    #install...
except:
    raise
