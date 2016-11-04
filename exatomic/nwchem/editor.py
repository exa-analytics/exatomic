# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
NWChem Editor
##################
"""
import numpy as np
import pandas as pd
from io import StringIO
from exatomic.container import Universe
from exatomic.editor import Editor as AtomicEditor
try:
    from exatomic.algorithms.basis import spher_lml_count, cart_lml_count, rlmap
except ImportError:
    from exatomic.algorithms.basis import spher_ml_count, cart_ml_count, lmap, lorder
    rlmap = {value: key for key, value in lmap.items() if len(key) == 1}
    spher_lml_count = {lorder.index(key): value for key, value in spher_ml_count.items()}
    cart_lml_count = {lorder.index(key): value for key, value in cart_ml_count.items()}


class Editor(AtomicEditor):
    """
    Base NWChem editor
    """
    def _expand_summary(self):
        '''
        Adds basis set information to the basis set summary table.
        Requires a parsed basis set object.
        '''
        if any('bas_' in col for col in self.basis_set_summary):
            return
                #lcounts = bfns.apply(lambda y: y['L'].values[0]).value_counts()
                #for l, lc in lcounts.items():
            #        lcounts[l] = lc * spher_lml_count[l] // cart_lml_count[l]
        #        lc = lcounts.sum()
        #rlmap = {value: key for key, value in lmap.items() if len(key) == 1}
        lmax = self.gaussian_basis_set['L'].cat.as_ordered().max()
        bs = self.gaussian_basis_set.groupby('set')
        bss = self.basis_set_summary
        pdata = []
        bdata = []
        cartcnt = []
        for seht in bss.index:
            cartcount = pd.Series([0] * len(bss.index))
            pdata.append([])
            bdata.append([])
            cartcnt.append(0)
            b = bs.get_group(seht)
            prims = b['L'].value_counts()
            bsfns = b.groupby('L').apply(lambda x: len(x['shell_function'].unique()))
            for i in range(lmax + 1):
                try:
                    pdata[-1].append(prims.ix[i])
                    bdata[-1].append(bsfns.ix[i])
                    cartcnt[-1] += bsfns.ix[i] * cart_lml_count[i]
                except KeyError:
                    pdata[-1].append(0)
                    bdata[-1].append(0)
        pdata = pd.DataFrame(pdata)
        bdata = pd.DataFrame(bdata)
        data = pd.concat([pdata, bdata], axis=1)
        data.index.name = 'set'
        sl = len(data.columns) // 2
        data.columns = ['prim_' + rlmap[i] for i in data.columns[:sl]] + \
                       ['bas_' + rlmap[i] for i in data.columns[sl:]]
        cartperatom = pd.Series(cartcnt)
        cartperatom.index.name = 'set'
        data['cart_per_atom'] = cartperatom
        data['cartesian_count'] = cartperatom * self.atom['set'].value_counts()
        self.basis_set_summary = pd.concat([bss, data], axis=1)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.meta is None:
            self.meta = {'program': 'nwchem'}
        else:
            self.meta.update({'program': 'nwchem'})
