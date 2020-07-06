# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Input Generator and Parser
###############################
"""
from .editor import Editor


_template = """\
{seward}

{calc}

{postcalc}
"""


_seward = """\
&SEWARD &END
Title
{title}
{atoms_bases}
{seward_options}
End of input
"""


_scf = """\
&SCF &END
Title
{title}
Charge
{charge}
{mult}
End of input
"""


_rasscf = """\
{rasrestart}
&RASSCF &END
Spin
{mult}
Symmetry
{symmetry}
nActEl
{nactel} {ras1_out_of} {ras3_into}
Inactive
{inactive}
Ras1
{ras1}
Ras2
{ras2}
Ras3
{ras3}
Iter
{iterations}
{restart_key}
CIRoots
{ciroots} {ciroots} 1
ORBAppear
{orbappear}
Outorbital
{outorbital}
levshift
{levshift}{alter}
End of input
"""


_caspt2 = """\
&CASPT2 &END
End of input
"""


_rassi = """\
&RASSI &END
End of input
"""


class InputGenerator(Editor):
    def __init__(self, *args, **kwargs):
        super(InputGenerator, self).__init__(*args, **kwargs)


def _handle_basis(basis, atom_types):
    if isinstance(basis, (str, dict)):
        return {atom: basis for atom in atom_types}
    if isinstance(basis, list):
        return {atom: bas for atom, bas in basis}
    raise TypeError("Cannot handle basis with type {}".format(type(basis)))


def write_molcas_input(universe, fp=None, seward=True, caspt2=False, rassi=False,
                       task='scf', title='', charge=0, seward_options=None,
                       basis='ANO-RCC-VDZP', mult=1, iterations=100, ras1=0,
                       inactive=0, alter='',
                       ras2=0, ras3=0, nactel=0, ras1_out_of=0, ras3_into=0,
                       ciroots=1, levshift=0, symmetry=1, restart='',
                       restart_key='', orbappear='compact', outorbital='canonical',):
    """
    Molcas is not a black box. Do not use this function.
    basis can be string, list of tuples or dictionary of symbol, basis pairs
    seward_options is an iterable of string options to go in seward
    if restart is specified and restart_key is not, restart is ignored
    """
    kwargs = {'seward': '', 'calc': '', 'postcalc': ''}
    if seward:
        cols = ['tag', 'x', 'y', 'z']
        atoms_bases = ''
        if 'tag' in universe.atom:
            pass
        else:
            tags = []
            tagcount = {}
            for atom in universe.atom['symbol']:
                tagcount.setdefault(atom, 0)
                tagcount[atom] += 1
                tags.append(atom + str(tagcount[atom]))
            universe.atom['tag'] = tags
        atom_types = universe.atom['symbol'].unique()
        basis = _handle_basis(basis, atom_types)
        for i, atom in enumerate(atom_types):
            atoms_bases += 'Basis set\n{}.{}\n'.format(atom, basis[atom])
            subatom = universe.atom[universe.atom['symbol'] == atom]
            atoms_bases += subatom[cols].to_string(index=None, header=None)
            if i == len(atom_types) - 1:
                atoms_bases += '\nEnd of basis'
            else:
                atoms_bases += '\nEnd of basis\n'
        sew_opts = ''
        if seward_options is not None:
            for i, string in enumerate(seward_options):
                if i == len(seward_options) - 1:
                    sew_opts += string
                else:
                    sew_opts += string + '\n'
        kwargs['seward'] = _seward.format(title=title, atoms_bases=atoms_bases,
                                          seward_options=sew_opts)

    totalz = universe.atom['Z'].sum()
    # Calculation type
    if task == 'scf':
        nopen = int(mult) - 1
        scfmult = 'UHF\nZSPIn\n{}'.format(nopen)
        kwargs['calc'] = _scf.format(title=title, charge=charge, mult=scfmult)
        # SCF check
        print('Check    :\n' \
              'Total Z  : {}\n' \
              'Total e- : {}\n' \
              'nopen orb: {}\n'.format(totalz, totalz - charge, nopen))
    elif task == 'rasscf':
        if restart_key == 'LUMORB':
            rasrestart = '>> COPY {} INPORB'.format(restart)
        elif restart_key == 'JOBIPH':
            rasrestart = '>> COPY {} JOBIPH'.format(restart)
        elif not restart_key:
            rasrestart = ''
        else:
            raise Exception('restart_key must be LUMORB or JOBIPH')
        if alter:
            altfmt = '\nALTER\n'
            if isinstance(alter, str):
                altfmt += alter
            elif isinstance(alter, list):
                for i, alt in enumerate(alter):
                    if i == len(alter) - 1:
                        altfmt += alt
                    else:
                        altfmt += alt + '\n'

        kwargs['calc'] = _rasscf.format(mult=mult, symmetry=symmetry, nactel=nactel,
                                        ras1_out_of=ras1_out_of, ras3_into=ras3_into,
                                        inactive=inactive, ras1=ras1, ras2=ras2,
                                        ras3=ras3, iterations=iterations,
                                        rasrestart=rasrestart, restart_key=restart_key,
                                        ciroots=ciroots, orbappear=orbappear,
                                        outorbital=outorbital, levshift=levshift,
                                        alter=altfmt)
        # RASSCF check
        sumelec = inactive * 2 + nactel
        print('Check     :\n'\
              'Spin      : {}\n' \
              'Total Z   : {}\n' \
              'Total e-  : {}\n' \
              'Inactives : {}\n' \
              'Ras1 holes: {}\n' \
              'Ras2 size : {}\n' \
              'Ras3 holes: {}\n' \
              'Sum of e- : {}'.format(mult, totalz, totalz - charge, inactive,
                                      ras1, ras2, ras3, sumelec))
        print('The specification of your ({}({},{}){}) (RAS1(NACT,RAS2)RAS3)' \
              'active space corresponds to a molecular charge of {}'.format(
              ras1, nactel, ras2, ras3, totalz - sumelec))
    else:
        raise Exception('task must be scf or rasscf')

    if caspt2:
        raise NotImplementedError("currently does not support caspt2")

    if rassi:
        raise NotImplementedError("currently does not support rassi")

    fl = InputGenerator(_template)
    if fp is not None:
        fl.write(fp, **kwargs)
    else:
        return fl.format(**kwargs)
