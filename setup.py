#!/usr/bin/env python
from setuptools import setup, find_packages
from exatomic import __version__


try:
    setup(
        name='exatomic',
        version=__version__,
        description='Computational chemistry functionality for exa',
        author='Tom Duignan & Alex Marchenko',
        author_email='exa.data.analytics@gmail.com',
        url='https://exa-analytics.github.io/exatomic',
        packages=find_packages(),
        package_data={'exatomic': ['_nbextensions/*.js']},
        include_package_data=True,
        license='Apache License Version 2.0'
    )
    #install...
except:
    raise
