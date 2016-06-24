# -*- coding: utf-8 -*-
'''
Configuration for exatomic
====================================
Additional configuration required for the exatomic package.
'''
import os
from exa import global_config as _conf
from exa.utility import mkp


pkg = os.path.dirname(__file__)
_conf['exatomic_nbext_localdir'] = mkp(pkg, '_nbextensions')
_conf['exatomic_nbext_sysdir'] = mkp(_conf['nbext_sysdir'], 'exatomic')
