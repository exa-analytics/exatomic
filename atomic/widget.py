# -*- coding: utf-8 -*-
'''
Widget (Universe)
======================
'''
from traitlets import Unicode
from exa.widget import ContainerWidget


class UniverseWidget(ContainerWidget):
    '''
    '''
    _view_module = Unicode('nbextensions/exa/atomic/universe').tag(sync=True)
    _view_name = Unicode('UniverseView').tag(sync=True)
