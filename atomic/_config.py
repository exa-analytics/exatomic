# -*- coding: utf-8 -*-
'''
Configuration for atomic
====================================
Additional configuration required for the atomic package.
'''
import os
from exa import _conf
from exa.utility import mkp


pkg = os.path.dirname(__file__)
_conf['atomic_nbext_localdir'] = mkp(pkg, 'nbextensions')
_conf['atomic_nbext_sysdir'] = mkp(_conf['nbext_sysdir'], 'atomic')
