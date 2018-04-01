.. Copyright (c) 2015-2018, Exa Analytics Development Team
.. Distributed under the terms of the Apache License 2.0

Installation
##############
This package makes heavy use of the Jupyter notebook and the Python programming
language. The easiest way to get started is via the Anaconda Python distribution.
Download and install [conda](https://conda.io/miniconda.html), and then follow
the instructions below.


Anaconda
#######################
Using anaconda or miniconda::

    conda install -c exaanalytics exatomic


Pypi
#######################
Using pip::

    sudo pip install exatomic
    sudo jupyter nbextension enable exatomic --py --sys-prefix


Repository
#########################
Manually (or for a development installation)::

    git clone https://github.com/exa-analytics/exatomic
    cd exa
    pip install .
    jupyter nbextension install exatomic --py --sys-prefix
    jupyter nbextension enable exatomic --py --sys-prefix


What's Next?
#####################
- Users should check out the :ref:`started-label`
- Contributors should check out the :ref:`dev-label`
- The :ref:`api-label` contains usage and extension examples, and developer notes


Note
########
Although an effort has been made to fully support legacy Python 2.7.x, some (optional) dependencies
may conflict on Windows environments.
