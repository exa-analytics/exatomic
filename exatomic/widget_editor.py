# -*- coding: utf-8 -*-
# Copyright (c) 2015-2018, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Universe Editor Widget
#########################
"""
from traitlets import Unicode
from exatomic.widget_base import ExatomicScene


class EditorScene(ExatomicScene):
    _model_name = Unicode("EditorSceneModel").tag(sync=True)
    _view_name = Unicode("EditorSceneView").tag(sync=True)
