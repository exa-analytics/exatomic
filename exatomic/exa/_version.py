# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
import os


with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "static", "version.txt"))) as f:
    __version__ = f.read().strip()
