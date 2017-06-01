# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Base Composer
#############################
A few executables in the Quantum ESPRESSO suite share aspects of their template.
Common features of provided by a base composer class. Generating input files
should be done using the appropriate classes, for example
:class:`~exatomic.qe.pw.composer.PWInput`.
"""
from exa.core.composer import Composer, ComposerMeta


_pwtemplate = """{n: = :control}
{n: = :system}
{n: = :electrons}
{n: = :ions}
{n: = :cell}
{atomic_species}
{atomic_positions}
{k_points}
{cell_parameters}
{occupations}
{constraints}
{atomic_forces}
"""


class PWCPComposer(Composer):
    """
    Base composer for Quantum ESPRESSO's cp.x and pw.x input files. Provides
    a few convience methods.
    """
    _key_4space = "    "
    _key_nl = "\n"
    _namelist_template = "&{name}\n{text}\n/"

    def add_psps(self, pspdct, massdct=None):
        """
        Given a dictionary of psp information, build the ``atomic_species`` attribute.

        The optional ``massdct`` dictionary can be used to modify atom masses. If no
        masses are provided default masses are used from the isotopes library.

        Args:
            pspdct (dict): Dictionary of atom name, pseudopotential filename pairs
            massdct (dict): Dictionary of atom name, atom mass pairs
        """
        pass

    def _compose_dict(self, val, **kwargs):
        """Dictionary keyword parameters are supposed to be 'namelists'."""
        text = super(PWCPComposer, self)._compose_dict(val, **kwargs)
        text = self._key_4space + text.replace(self._key_nl, self._key_nl+self._key_4space)
        return self._namelist_template.format(name=kwargs['name'], text=text)

    def _compose_str(self, val, **kwargs):
        """Cards require their text to start with the card name."""
        text = super(PWCPComposer, self)._compose_str(val, **kwargs)
        name = kwargs['name'].lower()
        if not text.lower().startswith(name):
            text = " ".join((name.upper(), text))
        return text
