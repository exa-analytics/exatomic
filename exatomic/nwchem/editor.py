# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
NWChem Editor
##################
"""
import numpy as np
<<<<<<< HEAD
#import pandas as pd
#from io import StringIO
#from exatomic.container import Universe
#from exatomic.editor import Editor as AtomicEditor
#from exatomic.algorithms.basis import spher_lml_count, cart_lml_count, rlmap
#
#class Editor(AtomicEditor):
#    """
#    Base NWChem editor
#    """
#    def __init__(self, *args, **kwargs):
#        super().__init__(*args, **kwargs)
#        if self.meta is None:
#            self.meta = {'program': 'nwchem'}
#        else:
#            self.meta.update({'program': 'nwchem'})
=======
import pandas as pd
from io import StringIO
from exatomic import Universe
from exatomic import Editor as AtomicEditor
from exatomic.algorithms.basis import spher_lml_count, cart_lml_count, rlmap


class Editor(AtomicEditor):
    """
    Base NWChem editor
    """
    def __init__(self, *args, **kwargs):
        super(Editor, self).__init__(*args, **kwargs)
        if self.meta is None:
            self.meta = {'program': 'nwchem'}
        else:
            self.meta.update({'program': 'nwchem'})
>>>>>>> 1c37655b6be3dca60b2adbeee8ca3767e5477943
