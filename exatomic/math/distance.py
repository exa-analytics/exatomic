# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Two Body Properties Computations
#####################################
Fast computations required for generating :class:`~exatomci.two.FreeTwo` and
:class:`~exatomic.two.PeriodicTwo` objects.

Warning:
    Without `numba`_, performance of this module is not guarenteed.

.. _numba: http://numba.pydata.org/
"""
import numpy as np
from exa._config import config
from exa.math.misc.repeat import repeat_counts_f8_2d
from exa.math.vector.cartesian import magnitude_xyz


def minimal_image(xyz, rxyz, oxyz):
    """
    """
    return np.mod(xyz, rxyz) + oxyz


def minimal_image_counts(xyz, rxyz, oxyz, counts):
    """
    """
    rxyz = repeat_counts_f8_2d(rxyz, counts)
    oxyz = repeat_counts_f8_2d(oxyz, counts)
    return minimal_image(xyz, rxyz, oxyz)


if config['dynamic']['numba'] == 'true':
    from numba import jit, vectorize
    from exa.math.vector.cartesian import magnitude_xyz
    types3 = ['int32(int32, int32, int32)', 'int64(int64, int64, int64)',
             'float32(float32, float32, float32)', 'float64(float64, float64, float64)']
    minimal_image_counts = jit(nopython=True, cache=True, nogil=True)(minimal_image_counts)
    minimal_image = vectorize(types3, nopython=True)(minimal_image)
