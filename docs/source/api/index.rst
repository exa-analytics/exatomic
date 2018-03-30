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
can be found at :ref:`examples-label` or via help::

    help(exatomic)             # Package help
    help(exatomic.isotopes)    # Module help
    help(exatomic.Universe)    # Class help
    exatomic.Universe?         # IPython environment

.. automodule:: exatomic.__init__
    :members:
    
.. automodule:: exatomic._version
    :members:

.. toctree::
    :maxdepth: 4
    :caption: Usage

    universe.rst
    editor.rst
    atom.rst
    two.rst
    orbital.rst
    basis.rst
    frame.rst
    molecule.rst


.. toctree::
    :maxdepth: 2
    :caption: Examples

    notebooks/nwchem.ipynb
    notebooks/qe.ipynb

.. toctree::
    :maxdepth: 2
    :caption: Extended API 

    matrices.rst
    algorithms/angles.rst
    algorithms/basis.rst

      angles.py*
      basis.py*
      car2sph.py*
      delocalization.py*
      diffusion.py*
      displacement.py*
      distance.py*
      geometry.py*
      harmonics.py*
      indexing.py*
      interpolation.py*
      neighbors.py*
      numerical.py*
      orbital.py*
      orbital_util.py*
      overlap.py*
      packing.py*
      pcf.py*
      slicing.py*
