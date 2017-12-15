# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Support for `NWChem`_
###########################

Note:
    This sub-package's API is organized in terms of the various 'tasks'
    supported by NWChem. A description of the tasks can be found on the NWChem
    `wiki`_.

.. _NWChem: www.nwchem-sw.org
.. _wiki: http://www.nwchem-sw.org/index.php/Release66:Top-level
"""
from . import scf, dft, nwpw
