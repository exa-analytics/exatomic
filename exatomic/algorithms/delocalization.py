# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
'''
Delocalization
################################
Miscellaneous functions for computing the delocalization error
inherent in carefully constructed exatomic universes. This universe
requires results from 3 different quantum chemical calculations on
an (N-1), N, and (N+1) electron system.
'''


def stack_unis(unis, same_frame=False):
    '''
    A non-general algorithm for combining "universes" into a single universe.
    '''
    attrs = [key[1:] for key, value in vars(unis[0]).items() if key not in
             ['_traits_need_update', '_widget', '_lines'] and key.startswith('_')]
    dfs = {}
    if same_frame:
        for attr in attrs:
            if attr == 'field':
                continue
            dfs[attr] = getattr(unis[0], attr)
        if hasattr(unis[0], 'field'):
            field_values = [i.field.field_values[0] for i in unis]
            field_params = pd.concat([pd.DataFrame(unis[0].field)] * len(unis)).reset_index(drop=True)
            field_params['frame'] = 0
            #field_params['label'] = field_params['label'].astype(np.int64)
            dfs['field'] = exatomic.field.AtomicField(field_params, field_values=field_values)
        return exatomic.Universe(**dfs)
    for attr in attrs:
        if attr == 'field':
            continue
        subdfs = []
        for i, uni in enumerate(unis):
            smdf = getattr(uni, attr)
            smdf['frame'] = i
            subdfs.append(smdf)
        adf = pd.concat(subdfs)
        adf.reset_index(drop=True, inplace=True)
        dfs[attr] = adf
    if hasattr(unis[0], 'field'):
        field_values = [i.field.field_values[0] for i in unis]
        field_params = pd.concat([pd.DataFrame(unis[0].field)] * len(unis)).reset_index(drop=True)
        field_params['frame'] = range(len(unis))
        field_params['label'] = 0
        field_params['label'] = field_params['label'].astype(np.int64)
        dfs['field'] = exatomic.field.AtomicField(field_params, field_values=field_values)
    return exatomic.Universe(**dfs)


class Deloc:
    @staticmethod
    def compute_deloc(uni, debug, jtype=None):
        #Basics
        cat = uni.frames[0]
        neut = uni.frames[1]
        an = uni.frames[2]
        cat_en = cat.energy
        neut_en = neut.energy
        an_en = an.energy
        #Electron counts
        aco = cat.orbitals.get_orbital()
        acate = cat.orbitals.get_orbital().orbital
        try:
            bco = cat.orbitals.get_orbital(spin=1)
        except IndexError:
            bco = aco
        ano = neut.orbitals.get_orbital()
        try:
            bno = neut.orbitals.get_orbital(spin=1)
        except IndexError:
            bno = ano
        aao = an.orbitals.get_orbital()
        try:
            bao = an.orbitals.get_orbital(spin=1)
        except IndexError:
            bao = aao

        #Eigenvalues
        if aco.orbital < ano.orbital:
            lumoca = cat.orbitals.get_orbital(index=aco.name + 1).energy
            homone = ano.energy
        if bco.orbital < bno.orbital:
            lumoca = cat.orbitals.get_orbital(index=bco.name + 1).energy
            homone = bno.energy
        if ano.orbital < aao.orbital:
            lumone = neut.orbitals.get_orbital(index=ano.name + 1).energy
            homoan = aao.energy
        if bno.orbital < bao.orbital:
            lumone = neut.orbitals.get_orbital(index=bno.name + 1).energy
            homoan = bao.energy

        #Compute J^2
        jone = homone + (cat_en - neut_en)
        jtwo = homoan + (neut_en - an_en)
        jtype = None
        j2 = jone ** 2 + jtwo ** 2
        if jtype == 'EA':
            j2 = jone ** 2
        elif jtype == 'IP':
            j2 = jtwo ** 2
        #Compute E(n) and curvature coefficients
        q = np.linspace(0, 1, 51)
        negdE = an_en - neut_en
        posdE = neut_en - cat_en
        autoev = 27.2114
        negE = (negdE*q + ((lumone - negdE)*(1 - q) + (negdE - homoan)*q)*q*(1 - q)) * autoev
        posE = (-posdE*q + ((lumoca - posdE)*(1 - q) + (posdE - homone)*q)*q*(1 - q)) * autoev
        ancur = (np.sum((lumone - negdE)*(6*q - 4) + (negdE - homoan)*(2 - 6*q)))/(len(q)*2) * autoev
        catcur = (np.sum((lumoca - posdE)*(6*q - 4) + (posdE - homone)*(2 - 6*q)))/(len(q)*2) * autoev
        data = np.empty((len(q)*2 - 1,), dtype = [('n', 'f8'), ('E', 'f8')])
        data['n'] = np.concatenate((np.fliplr([-q])[0][:-1], q))
        data['E'] = np.concatenate((np.fliplr([posE])[0][:-1], negE))
        #Proper object and tack on tidbits
        retvar = pd.DataFrame(data)
        retvar.ancur = ancur
        retvar.catcur = catcur
        retvar.j2 = j2
        retvar.name = uni.name
        retvar.focc = uni.focc
        if debug:
            print('============', uni.name, '============')
            print('aco = \n', aco)
            print('bco = \n', bco)
            if aco.orbital < ano.orbital:
                print('lumoca = \n', cat.orbitals.get_orbital(index=aco.name + 1))
            if bco.orbital < bno.orbital:
                print('lumoca = \n', cat.orbitals.get_orbital(index=bco.name + 1))
            print('ano = \n', ano)
            print('bno = \n', bno)
            if ano.orbital < aao.orbital:
                print('lumone = \n', neut.orbitals.get_orbital(index=ano.name + 1))
            if bno.orbital < bao.orbital:
                print('lumone = \n', neut.orbitals.get_orbital(index=bno.name + 1))
            print('aao = \n', aao)
            print('bao = \n', bao)
            print('lumoca energy = ', lumoca)
            print('homone energy = ', homone)
            print('lumone energy = ', lumone)
            print('homoan energy = ', homoan)
            print('cat energy = ', cat_en)
            print('neut energy = ', neut_en)
            print('an energy = ', an_en)
        return retvar

    def __new__(self, universe, debug=None, **kwargs):
        return self.compute_deloc(universe, debug=debug, **kwargs)
