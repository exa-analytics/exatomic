# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
'''
Exatomic Configuration
##########################
This module adds dynamic (only) configuration parameters. For complete
configuration options and usage see `exa`_.

.. _exa: https://github.com/exa-analytics/exa
'''
import os
import shutil
import atexit
from exa.utility import mkp
from exa._config import config, del_update


config['dynamic']['exatomic_pkgdir'] = os.path.dirname(__file__)


if config['paths']['update'] == '1':
    shutil.copyfile(mkp(config['dynamic']['exatomic_pkgdir'], '_static', 'exatomic_demo.ipynb'),
                    mkp(root, 'notebooks', 'exatomic_demo.ipynb'))
    atexit.register(del_update)
