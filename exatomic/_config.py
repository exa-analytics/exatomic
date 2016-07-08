# -*- coding: utf-8 -*-
'''
Exatomic Configuration
##########################
This module adds dynamic (only) configuration parameters. For complete
configuration options and usage see `exa`_.

.. _exa: https://github.com/exa-analytics/exa
'''
import os
from exa._config import config


config['dynamic']['exatomic_pkgdir'] = os.path.dirname(__file__)
