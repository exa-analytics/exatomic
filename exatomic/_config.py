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
import platform
from exa.utility import mkp
from exa._config import config


if platform.system().lower() == 'windows':   # Get exatomic's root directory
    home = os.getenv('USERPROFILE')
else:
    home = os.getenv('HOME')

root = mkp(home, '.exa', mk=True)            # Make exa root directory
config['dynamic']['exatomic_pkgdir'] = os.path.dirname(__file__)

if 'notebooks' in config['paths']['notebooks']:
    shutil.copyfile(mkp(config['dynamic']['exatomic_pkgdir'], '_static', 'exatomic_demo.ipynb'),
                    mkp(root, 'notebooks', 'exatomic_demo.ipynb'))
