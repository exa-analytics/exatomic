Getting Started
================
The nwchem package provides classes for handling NWChem input and output files.

As with all Python packages, exa can simply be imported inside any interpreter.

.. code-block:: Python

    import atomic
    from atomic import xyz
    uni = xyz.read_xyz('/path/to/file.xyz')
    type(uni) is atomic.Universe
