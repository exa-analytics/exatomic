# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Universe Notebook Widget
#########################
"""
import os
import subprocess
import pandas as pd
from glob import glob
from base64 import b64decode
from traitlets import Unicode
from ipywidgets import Layout
from exawidgets import ThreeScene, BaseBox
from exa.utility import mkp

# Default layouts
width = "600"
height = "450"
gui_lo = Layout(width="200px")

class TestUniverseScene(ThreeScene):
    """basic :class:`~exatomic.container.universe` scene."""
    _model_module = unicode("jupyter-exatomic").tag(sync=True)
    _view_module = unicode("jupyter-exatomic").tag(sync=True)
    _model_name = unicode("TestUniverseSceneModel").tag(sync=True)
    _view_name = unicode("TestUniverseSceneView").tag(sync=True)
    scn_clear = Bool(False).tag(sync=True)
    scn_saves = Bool(False).tag(sync=True)
    # Blah something about CaselessStrEnums
    kind = unicodde("null").tag(sync=True)
    field = unicode("null").tag(sync=True)
    field_nx = int(30).tag(sync=True)
    field_ny = int(30).tag(sync=True)
    field_nz = int(30).tag(sync=True)
    field_iso = float(0.005).tag(sync=True)

funcs = ['hwfs', 'gtos']
hwfs = ['1s',
        '2s',   '2px',  '2py', '2pz',
        '3s',   '3px',  '3py', '3pz',
        '3d-2', '3d-1', '3d0', '3d+1', '3d+2']
gtos = ['s', 'px', 'py', 'pz', 'd200', 'd110',
        'd101', 'd020', 'd011', 'd002']
funcs = {'hwfs': hwfs, 'gtos': gtos}
#shos = []

class TestUniverse(BaseBox):
    """Test :class:`~exatomic.container.Universe` test widget."""
    _model_module = Unicode("jupyter-exatomic").tag(sync=True)
    _view_module = Unicode("jupyter-exatomic").tag(sync=True)
    _model_name = Unicode("TestUniverseModel").tag(sync=True)
    _view_name = Unicode("TestUniverseView").tag(sync=True)
    scene = Instance(TestUniverseScene).tag(sync=True, **widget_serialization)

    def __init__(self, *args, **kwargs):
        scene = TestUniverse()
        scn_clear = Button(icon="bomb", description=" Clear", layout=gui_lo)
        scn_saves = Button(icon="camera", description=" Save", layout=gui_lo)
        field = Dropdown(options=field_options, layout=gui_lo)
        field_nx = IntSlider(description="N$_{x}$", **field_n_lims)
        field_ny = IntSlider(description="N$_{y}$", **field_n_lims)
        field_nz = IntSlider(description="N$_{z}$", **field_n_lims)
        field_iso = FloatSlider(min=3.0, max=10.0, description="Iso.", layout=gui_lo)
        # Button callbacks
        def _scn_clear(b): self.scene.scn_clear = not self.scene.scn_clear == True
        def _scn_saves(b): self.scene.scn_saves = not self.scene.scn_saves == True
        # Slider callbacks
        def _field(c): self.scene.field = c["new"]
        def _field_nx(c): self.scene.field_nx = c["new"]
        def _field_ny(c): self.scene.field_ny = c["new"]
        def _field_nz(c): self.scene.field_nz = c["new"]
        def _field_iso(c): self.scene.field_iso = c["new"]
        # Button handlers
        geo_shape.on_click(_geo_shape)
        scn_clear.on_click(_scn_clear)
        scn_saves.on_click(_scn_saves)
        geo_color.on_click(_geo_color)
        # Slider handlers
        field.observe(_field, names="value")
        field_nx.observe(_field_nx, names="value")
        field_ny.observe(_field_ny, names="value")
        field_nz.observe(_field_nz, names="value")
        field_iso.observe(_field_iso, names="value")
        # Put it all together
        # Labels separately
        gui = VBox([scn_clear, scn_saves, geo_shape, geo_color, field,
                    field_iso, field_nx, field_ny, field_nz])
        children = HBox([gui, scene])
        super(TestContainer, self).__init__(*args,
                                            children=[children],
                                            scene=scene,
                                            **kwargs)


class UniverseScene(TestUniverse):
    """basic :class:`~exatomic.container.universe` scene."""
    _model_name = unicode("UniverseSceneModel").tag(sync=True)
    _view_name = unicode("UniverseSceneView").tag(sync=True)


class Universe(TestUniverse):
    """Test :class:`~exatomic.container.Universe` widget."""
    _model_name = Unicode("UniverseModel").tag(sync=True)
    _view_name = Unicode("UniverseView").tag(sync=True)


# class UniverseWidget(ContainerWidget):
#     """
#     Custom widget for the :class:`~exatomic.universe.Universe` data container.
#     """
#     _model_module = Unicode("jupyter-exatomic").tag(sync=True)
#     _view_module = Unicode("jupyter-exatomic").tag(sync=True)
#     _model_name = Unicode("UniverseModel").tag(sync=True)
#     _view_name = Unicode('UniverseView').tag(sync=True)
#
#
#     def _handle_image(self, data):
#         savedir = os.getcwd()
#         if self.params['savedir'] != "":
#             savedir = self.params['save_dir']
#         if self.params['filename'] != "":
#             imgname = filename
#         else:
#             nxt = 0
#             try:
#                 lgfls = [fl.split(os.sep)[-1] for fl in glob(os.sep.join([savedir, "*png"]))]
#                 numbers = ["".join([c for c in fl if c.isdigit()]) for fl in lgfls]
#                 last = sorted(map(int, numbers))[-1]
#                 nxt = last + 1
#                 imgname = "{:06d}.png".format(nxt)
#             except:
#                 imgname = "{:06d}.png".format(nxt)
#         if os.path.isfile(os.sep.join([savedir, imgname])):
#             print("Automatic file name generation failed. Use uni._widget.params['filename']")
#             return
#         with open(os.sep.join([savedir, imgname]), "wb") as f:
#             f.write(b64decode(data.replace("data:image/png;base64,", "")))
#         # TODO : this likely won"t work on windows but SHOULD automatically
#         #        crop the image to minimize whitespace of the final image.
#         try:
#             crop = " ".join(["convert -trim", imgname, imgname])
#             subprocess.call(crop, cwd=savedir, shell=True)
#         except:
#             pass
