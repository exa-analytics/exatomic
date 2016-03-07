# -*- coding: utf-8 -*-
'''
Diffusion Coefficients
========================
Various algorithms for computing diffusion coefficients are coded here.
'''
from exa import _np as np
from exa import _pd as pd


def einstein_relation(universe, msd):
    '''
    Compute the (time dependent) diffusion coefficient using Einstein's relation.

    .. math::

        D\left(t\\right) = \\frac{1}{6Nt}\\sum_{i=1}^{N}\\left|\\mathbf{r}_{i}\left(t\\right)
            - \\mathbf{r}_{i}\\left(0\\right)\\right|^{2}
            
        D = \\lim_{t\\to\\infty} D\\left(t\\right)
    '''
    pass
