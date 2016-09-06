| |logo|
##################
Installation
##################
| |conda|
| |pypi|
Exatomic is available through `anaconda`_,

.. code-block:: bash

    conda install -c exaanalytics exatomic

or `pypi`_.

.. code-block:: bash

    pip install exatomic

After installation, run the following to set up visualization.

.. code-block:: bash

    exa -u

If using conda, make sure to be on up-to-date versions of `ipywidgets` and
`notebook`:

.. code-block:: bash

    conda install -c conda-forge notebook ipywidgets

Visualization currently support Chrome and Firefox.

###################
Getting Started
###################
| |docs|
| |gitter|
| |doi|
Documentation can be built using `sphinx`_:

.. code-block:: bash

    cd doc
    make html    # or .\make.bat html

##################
Status
##################
| |build|
| |issues|
| |cov|

###############
Legal
###############
| |lic|
| Copyright (c) 2015-2016, Exa Analytics Development Team
| Distributed under the terms of the Apache License 2.0

.. _anaconda: https://www.continuum.io/downloads
.. _pypi: https://pypi.python.org/pypi
.. _sphinx: http://www.sphinx-doc.org/en/stable/

.. |logo| image:: doc/source/_static/logo.png
    :target: doc/source/_static/logo.png
    :alt: Exatomic Analytics

.. |build| image:: https://travis-ci.org/exa-analytics/exatomic.svg?branch=master
    :target: https://travis-ci.org/exa-analytics/exatomic
    :alt: Build Status

.. |docs| image:: https://readthedocs.org/projects/exatomic/badge/?version=latest
    :target: http://exatomic.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |conda| image:: https://anaconda.org/exaanalytics/exatomic/badges/installer/conda.svg
    :target: https://conda.anaconda.org/exaanalytics
    :alt: Anaconda Version

.. |pypi| image:: https://badge.fury.io/py/exatomic.svg
    :target: https://badge.fury.io/py/exatomic
    :alt: PyPI Version

.. |gitter| image:: https://badges.gitter.im/exa-analytics/exatomic.svg
   :target: https://gitter.im/exa-analytics/exatomic?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge
   :alt: Join the chat at https://gitter.im/exa-analytics/exatomic

.. |issues| image:: https://www.quantifiedcode.com/api/v1/project/99e4f26905194100ad4c27aba432ec4c/badge.svg
  :target: https://www.quantifiedcode.com/app/project/99e4f26905194100ad4c27aba432ec4c
  :alt: Code issues

.. |cov| image:: https://coveralls.io/repos/github/exa-analytics/exatomic/badge.svg
    :target: https://coveralls.io/github/exa-analytics/exatomic
    :alt: Code Coverage

.. |lic| image:: http://img.shields.io/:license-apache-blue.svg?style=flat-square
    :target: http://www.apache.org/licenses/LICENSE-2.0
    :alt: License

.. |doi| image:: https://zenodo.org/badge/23807/exa-analytics/exatomic.svg
    :target: https://zenodo.org/badge/latestdoi/23807/exa-analytics/exatomic
    :alt: DOI
