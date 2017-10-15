<<<<<<< HEAD
# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Gaussian Editor
#######################
Base class on top of exatomic.Editor for NBO Editors
"""
#
#from exatomic import Editor as AtomicEditor
#
#class Editor(AtomicEditor):
#
#    def __init__(self, *args, **kwargs):
#        super().__init__(*args, **kwargs)
#        if self.meta is not None:
#            self.meta.update({'program': 'nbo'})
#        else:
#            self.meta = {'program': 'nbo'}
=======
# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Gaussian Editor
#######################
Base class on top of exatomic.Editor for NBO Editors
"""
from exatomic import Editor as AtomicEditor


class Editor(AtomicEditor):
    def __init__(self, *args, **kwargs):
        super(Editor, self).__init__(*args, **kwargs)
        if self.meta is not None:
            self.meta.update({'program': 'nbo'})
        else:
            self.meta = {'program': 'nbo'}
>>>>>>> 1c37655b6be3dca60b2adbeee8ca3767e5477943
