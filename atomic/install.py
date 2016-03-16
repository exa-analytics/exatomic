# -*- coding: utf-8 -*-
'''
Installer
====================
'''
import os
from notebook import install_nbextension
from exa.config import Config
from exa.install import install_notebook_widgets
from exa.utility import mkpath
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
