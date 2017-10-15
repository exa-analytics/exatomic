# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
NWChem Editor
##################
"""
import numpy as np
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
