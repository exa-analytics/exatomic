# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Exatomic Configuration
##########################
This module adds dynamic (only) configuration parameters. For complete
configuration options and usage see `exa`_.

.. _exa: https://github.com/exa-analytics/exa
"""
import os
import shutil
import atexit
from exa._config import config


def set_update():
    config['exatomic']['update'] = '1'

def del_update():
    config['exatomic']['update'] = '0'


config['dynamic']['exatomic_pkgdir'] = os.path.dirname(__file__)
if 'exatomic' not in config:
    config['exatomic'] = {'update': '1'}


if config['exatomic']['update'] == '1':
    key = 'root' if 'root' in config['dynamic'] else 'exa_root'
    shutil.copyfile(os.sep.join([config['dynamic']['exatomic_pkgdir'], '_static', 'exatomic_demo.ipynb']),
    os.sep.join([config['dynamic'][key], 'notebooks', 'exatomic_demo.ipynb']))
    shutil.copyfile(os.sep.join([config['dynamic']['exatomic_pkgdir'], '_static', 'porphyrin.xyz']),
    os.sep.join([config['dynamic'][key], 'data', 'examples', 'porphyrin.xyz']))
    atexit.register(del_update)
