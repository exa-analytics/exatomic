# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Gaussian Editor
#######################
Base class on top of exatomic.Editor for Gaussian Editors
"""
<<<<<<< HEAD
#
#import pandas as pd
#from exatomic import Editor as AtomicEditor
#
#class Editor(AtomicEditor):
#
#    def __init__(self, *args, **kwargs):
#        super().__init__(*args, **kwargs)
#        if self.meta is not None:
#            self.meta.update({'program': 'gaussian'})
#        else:
#            self.meta = {'program': 'gaussian'}
=======
import pandas as pd
from exatomic import Editor as AtomicEditor


class Editor(AtomicEditor):
    def __init__(self, *args, **kwargs):
        super(Editor, self).__init__(*args, **kwargs)
        if self.meta is not None:
            self.meta.update({'program': 'gaussian'})
        else:
            self.meta = {'program': 'gaussian'}
>>>>>>> 1c37655b6be3dca60b2adbeee8ca3767e5477943
