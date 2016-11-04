# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
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
import pandas as pd
import numpy as np
from exa.relational import Length as L
from exatomic.atom import Atom
from exatomic.container import Universe
from .editor import Editor


_template = """echo
start {{name}}
title {{title}}
charge {{charge}}

geometry {{geomopts}}
{{atom}}
end

basis {{basisopts}}
{{basis}}
end
{extras}
{calc}{{prop}}

task {{task}}"""


class InputGenerator(Editor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

def _handle_arg(opt, info):
    type1 = {'basis': 'library', 'ecp': 'library'}
    type2 = ['convergence']
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
        if type(info) != list:
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
        if type(info) is list:
            return ' '.join([item for item in info])
        else:
            print('{} keyword not handled correctly with value {}'.format(opt, info))


calcscf = """scf
 nopen {mult}
 maxiter {iterations}
end"""

calcdft = """dft
 direct
 mult {mult}
 grid xfine
 xc {xc}
 iterations {iterations}
{convergence}
 {restart}
end"""

#extras = """{{ecp}}{{relativistic}}{{tddft}}"""

def write_nwchem_input(universe, task='scf', fp=None, name=None, title=None,
                       charge=0, geomopts='units bohr\nsymmetry c1',
                       basisopts='spherical', basis='* library 6-31G',
                       mult=1, xc='b3lyp',
                       iterations=100, convergence='nolevelshifting',
                       # Block arguments
                       prop=' nbofile 2', relativistic='', tddft='',
                       ecp='',):
    """
    Due to the complexity of the NWChem program and the innumerable
    permutations of input file formats, this is in no way meant to be
    an exhaustive wrapper of NWChem input files. Alternatively,
    valid key words are handled according to the types of the
    arguments being passed to it. If the argument is a string, it should
    be formatted how you want with new line breaks (see default argument
    for geomopts). Multiple options for the same keyword are handled as
    lists of tuples (example: basis=[('C', '3-21G'), ('H', '6-31G**')]).
    Similarly, convergence criteria may be specified with convergence =
    ['nolevelshifting', 'ncydp 30', 'damp 70']. The closer your string
    formatting is to what NWChem expects, the less likely it is that you
    will obtain syntax errors in the written input file.
    """
    calc = calcdft if task == 'dft' else calcscf
    extras = ''
    extradict = {}
    for arg, extra in [('ecp', ecp), ('property', prop),
                       ('relativistic', relativistic), ('tddft', tddft)]:
        if extra:
            extras += '{' + arg + '}'
            extradict[arg] = _handle_arg(arg, extra)
    fl = InputGenerator(_template.format(calc=calc, extras=extras))
    keys = [key.split('}')[0].split(':')[0] for key in _template.split('{')[1:]]
    keys += [key.split('}')[0].split(':')[0] for key in calcscf.split('{')[1:]]
    keys += [key.split('}')[0].split(':')[0] for key in calcdft.split('{')[1:]]
    kwargs = {key: '' for key in keys}

    atom = universe.atom[['symbol', 'x', 'y', 'z']]
    kwargs['atom'] = atom.to_string(index=None, header=None)
    if universe.name is not None:
        kwargs['name'] = universe.name
    elif name is not None:
        kwargs['name'] = name
    else:
        kwargs['name'] = ''.join(atom['symbol'])
    print(kwargs['name'])
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
    kwargs['convergence'] = _handle_arg('convergence', convergence)
    kwargs['task'] = task
    if prop and 'property' not in task:
        kwargs['task'] += ' property'
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
        fl.write(fp)
    else:
        return fl
