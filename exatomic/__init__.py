# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
A unified data platform for computational and theoretical chemists and
physicists.

Supported software
####################
- :mod:`~exatomic.adf.__init__`: `Amsterdam Density Functional`_

.. _Amsterdam Density Functional: https://www.scm.com
"""
def _jupyter_nbextension_paths():
    """Jupyter notebook extension directory paths."""
    return [{
        'section': "notebook",
        'src': "_nbextension",
        'dest': "jupyter-exatomic",
        'require': "jupyter-exatomic/extension"
    }]


from exatomic._version import __version__
from exatomic.xyz import XYZ
