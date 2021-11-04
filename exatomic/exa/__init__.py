# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Exa
#########
"""
import os
import tempfile
import logging.config
import yaml


with open(os.path.join(os.path.dirname(__file__),
          'conf', 'logging.yml'), 'r') as f:
    _log = yaml.safe_load(f.read())
_log['handlers']['file']['filename'] = os.path.join(tempfile.gettempdir(), 'exa.log')
logging.config.dictConfig(_log)

from .core import DataFrame, Series, Field3D, Field, Editor, Container, TypedMeta
from ._version import __version__
