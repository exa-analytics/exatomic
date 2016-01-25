# -*- coding: utf-8 -*-
'''
Tools
====================
Require internal imports.
'''
import os
from operator import itemgetter
from notebook import install_nbextension
from exa.config import Config
from exa.tools import install_notebook_widgets
from exa.utils import mkpath
from exa.errors import MissingColumns
from exa.relational.base import create_all


atomic_pkg = os.path.dirname(__file__)
atomic_static = mkpath(atomic_pkg, 'static')
atomic_nbext = mkpath(atomic_static, 'nbextensions')
atomic_extensions = mkpath(Config.extensions, 'atomic', mkdir=True)
atomic_config = [
    ('pkg', atomic_pkg),
    ('static', atomic_static),
    ('nbext', atomic_nbext),
    ('extensions', atomic_extensions)
]


def update_config():
    '''
    Update exa's configuration to include
    '''
    if not hasattr(Config, 'atomic'):
        Config['atomic'] = {}
    for k, v in atomic_config:
        if k not in Config['atomic']:
            Config['atomic'][k] = v
    Config.save()


def finalize_install(verbose=False):
    '''
    Run after installing this package.
    '''
    update_config()
    create_all()
    install_notebook_widgets(Config.atomic['nbext'],
                             Config.atomic['extensions'],
                             verbose=verbose)


def check(universe):
    '''
    '''
    rfc = ['rx', 'ry', 'rz', 'ox', 'oy', 'oz']    # Required columns in the Frame table for periodic calcs
    if 'periodic' in universe.frames.columns:
        if any(universe.frames['periodic'] == True):
            missing = set(rfc).difference(universe.frames.columns)
            if missing:
                raise MissingColumns(missing, universe.frames.__class__.__name__)
            return True
    return False


def formula_dict_to_string(fdict):
    '''
    '''
    return ''.join([k + '(' + str(fdict[k]) + ')' for k in sorted(fdict, key=itemgetter(0))])
