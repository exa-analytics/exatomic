# -*- coding: utf-8 -*-
'''
Installer
##################
'''
from exa._install import install as exinstall, install_notebook_widgets
from exatomic._config import config


def install(persist=False, verbose=False):
    '''
    '''
    exinstall(persist, verbose)
    install_notebook_widgets(config['exatomic_nbext_localdir'], config['exatomic_nbext_sysdir'], verbose)
