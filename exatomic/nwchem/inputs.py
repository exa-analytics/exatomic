# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Input Generator and Parser
#############################
Every attempt is made to follow the Documentation on the
NWChem `website`_ with a general theme of the input Generator
accepting keyword arguments mirroring the keywords accepted by
NWChem and values corresponding to the parameters in a calculation.

.. _website: http://www.nwchem-sw.org/index.php/Release66:NWChem_Documentation
"""
# """
# Due to the complexity of the NWChem program and the innumerable
# permutations of input file formats, this is in no way meant to be
# an exhaustive wrapper of NWChem input files. Alternatively,
# valid key words are handled according to the types of the
# arguments being passed to it. If the argument is a string, it should
# be formatted how you want with new line breaks (see default argument
# for geomopts). Multiple options for the same keyword are handled as
# lists of tuples (example: basis=[('C', '3-21G'), ('H', '6-31G**')]).
# Similarly, convergence criteria may be specified with convergence =
# ['nolevelshifting', 'ncydp 30', 'damp 70']. The closer your string
# formatting is to what NWChem expects, the less likely it is that you
# will obtain syntax errors in the written input file.
# """
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
#import pandas as pd
#import numpy as np
from .editor import Editor
#from exa.util.units import Length as L
#from exatomic import Universe


_template = """echo
start {{name}}
title {{title}}
charge {{charge}}

{{memory}}

geometry {{geomopts}}
{{atom}}
end

basis {{basisopts}}
{{basis}}
end

{{set}}
{extras}
{calc}{{prop}}
{{task}}"""


_calcscf = """scf
 nopen {mult}
 maxiter {iterations}
end"""

_calcdft = """dft
 direct
 mult {mult}
 xc {xc}
 iterations {iterations}
 tolerances {tolerances}
{grid}
{convergence}
 {dft_other}
 {restart}
end"""


class Input(Editor):

    @classmethod
    def from_universe(cls, uni, task='scf', fp=None, name=None, title=None,
                      charge=0, geomopts='units bohr\nsymmetry c1',
                      basisopts='spherical', basis='* library 6-31G',
                      mult=1, xc='b3lyp', iterations=100,
                      convergence='nolevelshifting', prop=' nbofile 2',
                      relativistic='', tddft='', ecp='', sets=None, tasks='property',
                      dft_other='', grid='xfine', tolerances='tight', memory=''):
        calc = _calcdft if task == 'dft' else _calcscf
        extras = ''
        extradict = {}
        for arg, extra in [('ecp', ecp), ('property', prop),
                           ('relativistic', relativistic), ('tddft', tddft)]:
            if extra:
                extras += '{' + arg + '}'
                extradict[arg] = _handle_arg(arg, extra)
        fl = cls(_template.format(calc=calc, extras=extras))
        keys = [key.split('}')[0].split(':')[0] for key in _template.split('{')[1:]]
        keys += [key.split('}')[0].split(':')[0] for key in _calcscf.split('{')[1:]]
        keys += [key.split('}')[0].split(':')[0] for key in _calcdft.split('{')[1:]]
        kwargs = {key: '' for key in keys}
        kwargs['atom'] = uni.atom.to_xyz()[:-1]
        if name is not None:
            kwargs['name'] = name
        else:
            kwargs['name'] = ''.join(atom['symbol'])
        kwargs['title'] = title if title is not None else kwargs['name']
        kwargs['charge'] = charge
        kwargs['geomopts'] = _handle_arg('geomopts', geomopts)
        kwargs['basisopts'] = _handle_arg('basisopts', basisopts)
        kwargs['basis'] = _handle_arg('basis', basis)
        if task == 'dft':
            kwargs['mult'] = mult
        elif mult - 1 > 0:
            kwargs['mult'] = str(mult - 1) + '\n uhf'
        else:
            kwargs['mult'] = mult - 1
        kwargs['xc'] = xc
        kwargs['iterations'] = iterations
        kwargs['tolerances'] = tolerances
        kwargs['convergence'] = _handle_arg('convergence', convergence)
        kwargs['grid'] = _handle_arg('grid', grid)
        kwargs['dft_other'] = _handle_arg('dft_other', dft_other)
        kwargs['memory'] = memory
        if sets != None:
            kwargs['set'] = _handle_arg('set', sets)
        kwargs['task'] = ''
        if isinstance(tasks, list):
            for i in tasks:
                kwargs['task'] += '\ntask '+task+' '+i
        else:
            kwargs['task'] += 'task '+task+' '+tasks
        #extras = {'ecp': _handle_arg('ecp', ecp),
        #          'tddft': _handle_arg('tddft', tddft),
        #          'property': _handle_arg('property', prop),
        #          'relativistic': _handle_arg('relativistic', relativistic)}
        kwargs.update(extradict)
    #### TASK AND EXTRAS

        #kwargs['prop'] = '\n\nproperty\n nbofile 2\nend'
        #kwargs['task'] = 'property'
        #kwargs['calc'] = calc
        #if options is not None:
        #    for opt, info in options.items():
        #        if opt in extras:
        #            _handle_info(opt, info, extras)
        #        elif kind == 'scf' and opt == 'mult':
        #            kwargs['mult'] = str(int(info) - 1) + '\n uhf' if int(info) > 1 else info
        #        else:
        #            _handle_info(opt, info, kwargs)
        #extras = ['\n' + key + '\n' + val for key, val in extras.items() if val]
        #kwargs['extras'] = '\n'.join([extra + '\nend' for extra in extras])
        fl.format(inplace=True, **kwargs)
        if fp is not None:
            if name is not None:
                fl.write(fp+name)
            else:
                fl.write(fp)
        else:
            return fl

    def __init__(self, *args, **kwargs):

        super(Input, self).__init__(*args, **kwargs)


def _handle_arg(opt, info):
    type1 = {'basis': 'library', 'ecp': 'library'}
    type2 = ['convergence', 'set', 'grid']
    type3 = ['ecp', 'property', 'tddft', 'relativistic']
    if isinstance(info, str):
        if opt in type3:
            return '\n{0}\n{1}\n{2}\n'.format(opt, info, 'end')
        return info
    if opt in type1:
        ret = ''
        for i, tup in enumerate(info):
            if i == len(info) - 1:
                ret = ' '.join([ret, tup[0], type1[opt], tup[1]])
            else:
                ret = ' '.join([ret, tup[0], type1[opt], tup[1], '\n'])
        if opt in type3:
            return '\n{0}\n{1}\n{2}\n'.format(opt, ret, 'end')
        return ret
    elif opt in type2:
        ret = ''
        if not isinstance(info, list):
            info = [info]
        for i, arg in enumerate(info):
            if i == len(info) - 1:
                ret = ' '.join([ret, opt, arg])
            else:
                ret = ' '.join([ret, opt, arg, '\n'])
        if opt in type3:
            return '\n{0}\n{1}\n{2}\n'.format(opt, ret, 'end')
        return ret
    else:
        if isinstance(info, list):
            return ' '.join([item for item in info])
        else:
            print('{} keyword not handled correctly with value {}'.format(opt, info))


def tuning_inputs(uni, name, mult, charge, basis, gammas, alphas,
                  route=None, link0=None,
                  field=None, writedir=None, deep=False):
    """
    Provided a universe, generate input files for functional tuning.
    Includes input keywords for orbital visualization within exatomic.
    Assumes you will copy restart checkpoint files to have the same
    names as the input files.

    Args
        uni (exatomic.container.Universe): molecular specification
        name (str): prefix for job names
        mult (int): spin multiplicity
        charge (int): charge of the system
        basis (list): tuples of atomic symbol, string of basis name
        gammas (iter): values of range separation parameter (omega)
        alphas (iter): fractions of Hartree-Fock in the short range
        route (list): strings or tuples of keyword, value pairs (default [("Pop", "full")])
        link0 (list): strings or tuples of keyword, value pairs
        writedir (str): directory path to write input files

    Returns
        editors (list): input files as exa.Editors
    """
    if route is None:
        route = [("Pop", "full")]
    fnstr = 'xcampbe96 1.0 cpbe96 1.0 HFexch 1.0\n'\
            ' cam {gam:.4f} cam_alpha {alp:.4f} cam_beta {bet:.4f}'.format
    jbnm = '{name}-{{gam:.2f}}-{{alp:.2f}}-{{chg}}'.format(name=name).format
    chgnms = ['cat', 'neut', 'an']
    chgs = [charge + 1, charge, charge - 1]
    mults = [2, 1, 2] if mult == 1 else [mult - 1, mult, mult + 1]
    fls = []
    for gam in gammas:
        for alp in alphas:
            #bet = 1 - alp
            for chgnm, chg, mult in zip(chgnms, chgs, mults):
                fnc = fnstr(gam=gam, alp=alp, bet=1-alp)
                jnm = jbnm(gam=gam, alp=alp, bet=1-alp, chg=chgnm)
                opts = {'charge': chg, 'mult': mult, 'task': 'dft',
                        'title': jnm, 'name': jnm, 'xc': fnc,
                        'basis': basis, 'prop': ''} #, 'writedir': writedir}
                fls.append(Input.from_universe(uni, **opts))
                fls[-1].name = jnm + '.nw'
    return fls

# def tuning_inputs(uni, name, mult, charge, basis, gammas, alphas,
#                   route=[('Pop', 'full')], link0=None, nproc=4, mem=4,
#                   field=None, writedir=None, deep=False):
    # def from_universe(cls, uni, task='scf', fp=None, name=None, title=None,
    #                   charge=0, geomopts='units bohr\nsymmetry c1',
    #                   basisopts='spherical', basis='* library 6-31G',
    #                   mult=1, xc='b3lyp', iterations=100,
    #                   convergence='nolevelshifting', prop=' nbofile 2',
    #                   relativistic='', tddft='', ecp=''):
