.. Copyright (c) 2015-2017, Exa Analytics Development Team
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

    import exatomic
    help(exatomic)             # Package help
    help(exatomic.isotopes)    # Module help
    help(exatomic.Universe)    # Class help
    exatomic.Universe?         # In an IPython environment (including the Jupyter notebook)

.. automodule:: exatomic.__init__
    :members:

.. automodule:: exatomic._version
    :members:

.. toctree::
    :maxdepth: 2
    :caption: Core API

    api/universe.rst


########################
Unittest Docs
########################
Source code of tests can sometimes provide useful information for developers
and users.
