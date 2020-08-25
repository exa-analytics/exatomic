# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Diffusion Coefficients
##########################
Various algorithms for computing diffusion coefficients are coded here.
"""
from exa.util.units import Length, Time
from exatomic.algorithms.displacement import absolute_squared_displacement


def einstein_relation(universe, input_time='ps', input_length='au',
                      length='cm', time='s'):
    """
    Compute the (time dependent) diffusion coefficient using Einstein's relation.

    .. math::

        D\left(t\\right) = \\frac{1}{6Nt}\\sum_{i=1}^{N}\\left|\\mathbf{r}_{i}\left(t\\right)
            - \\mathbf{r}_{i}\\left(0\\right)\\right|^{2}

        D = \\lim_{t\\to\\infty} D\\left(t\\right)

    Args:
        universe (:class:`~exatomic.core.universe.Universe`): The universe object
        input_time (str): String unit of 'time' column in frame table
        input_length (str): String unit of xyz coordinates
        length (str): String unit name of output length unit
        time (str): Sting unit name of output time unit

    Returns:
        d (:class:`~exa.core.numerical.DataFrame`): Diffussion coefficient as a function of time

    Note:
        The asymptotic value of the returned variable is the diffusion coefficient.
        The default units of the diffusion coefficient are :math:`\\frac{cm^{2}}{s}`.
    """
    msd = absolute_squared_displacement(universe).mean(axis=1)
    t = universe.frame['time'] * Time[input_time, time]
    msd *= Length[input_length, length]**2
    return msd/(6*t)
