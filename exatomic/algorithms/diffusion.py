# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Diffusion Coefficients
##########################
Various algorithms for computing diffusion coefficients are coded here.
"""
from exa.util.units import Length, Time
from exatomic.algorithms.displacement import absolute_squared_displacement


def einstein_relation(universe, length='cm', time='s'):
    """
    Compute the (time dependent) diffusion coefficient using Einstein's relation.

    .. math::

        D\left(t\\right) = \\frac{1}{6Nt}\\sum_{i=1}^{N}\\left|\\mathbf{r}_{i}\left(t\\right)
            - \\mathbf{r}_{i}\\left(0\\right)\\right|^{2}

        D = \\lim_{t\\to\\infty} D\\left(t\\right)

    Args:
        universe (:class:`~exatomic.Universe`): The universe object
        msd (:class:`~exa.DataFrame`): Mean squared displacement dataframe

    Returns:
        D (:class:`~exa.DataFrame`): Diffussion coefficient as a function of time

    Note:
        The asymptotic value of the returned variable is the diffusion coefficient.
        The default units of the diffusion coefficient are :math:`\\frac{cm^{2}}{s}`.
    """
    msd = absolute_squared_displacement(universe).mean(axis=1)
    t = universe.frame['time'].values
    msd *= Length['au', length]**2
    t /= Time['au', time]
    return msd / (6 * t)
