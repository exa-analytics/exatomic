.. Copyright (c) 2015-2018, Exa Analytics Development Team
.. Distributed under the terms of the Apache License 2.0

.. _api-label:

########################
User Docs
########################
The following sections describe syntax and usage of the functions and classes
provided by the Exa package. Documentation is organized for the typical use case;
a collection of structure text files need to be parsed into Pythonic data objects
and then organized into a container to facilitate visualization. Useful examples 
and syntax can be found in the Examples section (Jupyte notebooks) and via help::

    help(exatomic)             # Package help
    help(exatomic.isotopes)    # Module help
    help(exatomic.Universe)    # Class help
    exatomic.Universe?         # IPython environment

.. automodule:: exatomic.__init__
    :members:
    
.. automodule:: exatomic._version
    :members:

.. toctree::
    :maxdepth: 2
    :caption: Core Usage

    universe.rst
    editor.rst
    widgets.rst
    interfaces.rst
    atom.rst
    two.rst
    orbital.rst
    basis.rst
    tensor.rst
    frame.rst
    molecule.rst

.. toctree::
    :maxdepth: 2
    :caption: Extended API 

    field.rst
    matrices.rst
    algorithms/basis.rst
    algorithms/orbital.rst
    algorithms/integrals.rst
    algorithms/structural.rst
    algorithms/builders.rst
    algorithms/delocalization.rst
    algorithms/interpolation.rst
    algorithms/neighbors.rst

.. toctree::
    :maxdepth: 2
    :caption: Misc

    error.rst
    algorithms/indexing.rst
    tests/base_tests.rst
    tests/core_tests.rst
    tests/algorithms_tests.rst
