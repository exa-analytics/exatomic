#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
from setuptools import setup, find_packages
from exatomic import __version__

with open('README.rst') as f:
    description = f.read()
with open('requirements.txt') as f:
    dependencies = f.readlines()

setup(
    name='exatomic',
    version=__version__,
    description='A unified platform for computational chemists.',
    long_description=description,
    author='Tom Duignan, Alex Marchenko',
    author_email='exa.data.analytics@gmail.com',
    maintainer_email='exa.data.analytics@gmail.com',
    url='https://exa-analytics.github.io',
    download_url = 'https://github.com/exa-analytics/exatomic/tarball/v{}'.format(__version__),
    packages=find_packages(),
    package_data={'exatomic': ['_static/*.json', '_nbextension/*.js']},
    include_package_data=True,
    install_requires=dependencies,
    license='Apache License Version 2.0',
    keywords='analytics visualization computational chemistry',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Web Environment',
        'Framework :: IPython',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering :: Chemistry'
    ]
)

from exatomic._config import config, save
config['db']['update'] = '1'
config['js']['update'] = '1'
save()
