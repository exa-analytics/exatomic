# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Universe Notebook Widget
#########################
"""
import os
import subprocess
#import pandas as pd
from glob import glob
from base64 import b64decode
from traitlets import Unicode
from exawidgets.widget import BaseDOM
#from exawidgets import ContainerWidget
#from exa.utility import mkp
from traitlets import Unicode, Bool, Int, Float, Instance
from ipywidgets import (Layout, widget_serialization, Button,
                        Dropdown, register, VBox, HBox, FloatSlider)
from exawidgets import ThreeScene, BaseBox
from exa.utility import mkp

# Default layouts
width = "600"
height = "450"
gui_lo = Layout(width="200px")

class UniverseWidget(BaseDOM):
    """
    Custom widget for the :class:`~exatomic.universe.Universe` data container.
    """
    _model_module = Unicode("jupyter-exatomic").tag(sync=True)
    _view_module = Unicode("jupyter-exatomic").tag(sync=True)
    _model_name = Unicode("UniverseModel").tag(sync=True)
    _view_name = Unicode('UniverseView').tag(sync=True)


    def _handle_image(self, data):
        print("widget._handle_image")
        savedir = os.getcwd()
        if self.params['save_dir'] != "":
            savedir = self.params['save_dir']
        #if self.params['file_name'] != "":
        #    imgname = "none" #filename
        else:
            nxt = 0
            if self.params['filename'] != "":
                imgname = filename
                if not imgname.endswith(".png"):
                    imgname += ".png"
            else:
                imgname = "{:06d}.png".format(nxt)
                while os.path.isfile(os.sep.join([savedir, imgname])):
                    nxt += 1
                    imgname = "{:06d}.png".format(nxt)
            with open(os.sep.join([savedir, imgname]), "wb") as f:
                f.write(b64decode(data.replace("data:image/png;base64,", "")))
        # TODO : this likely won"t work on windows but SHOULD automatically
        #        crop the image to minimize whitespace of the final image.
        try:
            crop = " ".join(["convert -trim", imgname, imgname])
            subprocess.call(crop, cwd=savedir, shell=True)
        except:
            pass
@register("exatomic.UniverseScene")
class UniverseScene(ThreeScene):
    """:class:`~exatomic.container.universe` scene."""
    _model_module = Unicode("jupyter-exatomic").tag(sync=True)
    _view_module = Unicode("jupyter-exatomic").tag(sync=True)
    _model_name = Unicode("UniverseSceneModel").tag(sync=True)
    _view_name = Unicode("UniverseSceneView").tag(sync=True)


@register("exatomic.Universe")
class Universe(BaseBox):
    """:class:`~exatomic.container.Universe` widget."""
    _model_module = Unicode("jupyter-exatomic").tag(sync=True)
    _view_module = Unicode("jupyter-exatomic").tag(sync=True)
    _model_name = Unicode("UniverseModel").tag(sync=True)
    _view_name = Unicode("UniverseView").tag(sync=True)
    scene = Instance(UniverseScene).tag(sync=True, **widget_serialization)


@register("exatomic.TestUniverseScene")
class TestUniverseScene(UniverseScene):
    """Test :class:`~exatomic.container.universe` scene."""
    _model_name = Unicode("TestUniverseSceneModel").tag(sync=True)
    _view_name = Unicode("TestUniverseSceneView").tag(sync=True)
    scn_clear = Bool(False).tag(sync=True)
    scn_saves = Bool(False).tag(sync=True)
    field_iso = Float(0.005).tag(sync=True)
    # Blah something about CaselessStrEnums
    #kind = Unicode("null").tag(sync=True)
    #field = Unicode("null").tag(sync=True)
    #field_nx = Int(30).tag(sync=True)
    #field_ny = Int(30).tag(sync=True)
    #field_nz = Int(30).tag(sync=True)

funcs = ['hwfs', 'gtos']
hwfs = ['hydrogenic', '1s',
        '2s',   '2px',  '2py', '2pz',
        '3s',   '3px',  '3py', '3pz',
        '3d-2', '3d-1', '3d0', '3d+1', '3d+2']
gtos = ['gaussian', 's', 'px', 'py', 'pz', 'd200', 'd110',
        'd101', 'd020', 'd011', 'd002']

funcs = {'hwfs': hwfs, 'gtos': gtos}
#shos = []


@register("exatomic.TestUniverse")
class TestUniverse(Universe):
    """Test :class:`~exatomic.container.Universe` test widget."""
    _model_name = Unicode("TestUniverseModel").tag(sync=True)
    _view_name = Unicode("TestUniverseView").tag(sync=True)
    scene = Instance(TestUniverseScene).tag(sync=True, **widget_serialization)

    def __init__(self, *args, **kwargs):
        scene = TestUniverseScene()
        scn_clear = Button(icon="bomb", description=" Clear", layout=gui_lo)
        scn_saves = Button(icon="camera", description=" Save", layout=gui_lo)
        # field = Dropdown(options=field_options, layout=gui_lo)
        # field_nx = IntSlider(description="N$_{x}$", **field_n_lims)
        # field_ny = IntSlider(description="N$_{y}$", **field_n_lims)
        # field_nz = IntSlider(description="N$_{z}$", **field_n_lims)
        field_iso = FloatSlider(min=0.0001, max=0.1, step=0.0001, description="Iso.", layout=gui_lo)
        field_hyd = Dropdown(options=hwfs, layout=gui_lo)
        field_gau = Dropdown(options=gtos, layout=gui_lo)
        # # Button callbacks
        def _scn_clear(b): self.scene.scn_clear = not self.scene.scn_clear == True
        def _scn_saves(b): self.scene.scn_saves = not self.scene.scn_saves == True
        # # Slider callbacks
        def _field_hyd(c): self.scene.field_hyd = c["new"]
        def _field_gau(c): self.scene.field_gau = c["new"]
        # def _field_nx(c): self.scene.field_nx = c["new"]
        # def _field_ny(c): self.scene.field_ny = c["new"]
        # def _field_nz(c): self.scene.field_nz = c["new"]
        def _field_iso(c): self.scene.field_iso = c["new"]
        # # Button handlers
        # geo_shape.on_click(_geo_shape)
        # scn_clear.on_click(_scn_clear)
        # scn_saves.on_click(_scn_saves)
        # geo_color.on_click(_geo_color)
        # # Slider handlers
        # field.observe(_field, names="value")
        # field_nx.observe(_field_nx, names="value")
        # field_ny.observe(_field_ny, names="value")
        # field_nz.observe(_field_nz, names="value")
        field_iso.observe(_field_iso, names="value")
        # # Put it all together
        # # Labels separately
        gui = VBox([scn_clear, scn_saves, field_hyd, field_gau, field_iso])
                #, geo_shape, geo_color, field,
                 #field_iso, field_nx, field_ny, field_nz])
        children = HBox([gui, scene])
        super(TestUniverse, self).__init__(*args,
                                           children=[children],
                                           scene=scene,
                                           **kwargs)
