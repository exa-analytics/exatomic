<<<<<<< HEAD:exatomic/widget_small.py
## -*- coding: utf-8 -*-
## Copyright (c) 2015-2016, Exa Analytics Development Team
## Distributed under the terms of the Apache License 2.0
#"""
#Universe Notebook Widget
##########################
#"""
#
####
#### OLDER CODE KEPT HERE FOR TESTING PURPOSES
####   AND TO COMPARE TIMINGS WITH VARIOUS
####   MESSAGE PASSING CONVENTIONS BETWEEN
####         PYTHON AND JAVASCRIPT
####      ** NOT IN PRODUCTION USE **
####
#import os
#import numpy as np
#import pandas as pd
#from glob import glob
#from base64 import b64decode
#from collections import OrderedDict
#from traitlets import Bool, Float, Int, Instance, Unicode, List, Dict, Any, link
#from ipywidgets import (Widget, DOMWidget, Box, widget_serialization, Layout,
#                        Button, Dropdown, VBox, HBox, FloatSlider, IntSlider,
#                        register, Play, jslink, Checkbox)
### Imports expected to break
#from exatomic.container import Universe
#from exa.relational.isotope import symbol_to_radius, symbol_to_color
#
#
####################
## Default layouts #
####################
#
#width = "400"
#height = "400"
#gui_lo = Layout(width="195px")
#
#
########################
## Common GUI patterns #
########################
#
#uni_field_lists = OrderedDict([
#    ("Hydrogenic", ['1s',   '2s',   '2px', '2py', '2pz',
#                    '3s',   '3px',  '3py', '3pz',
#                    '3d-2', '3d-1', '3d0', '3d+1', '3d+2']),
#    ("Gaussian", ['s', 'px', 'py', 'pz', 'd200', 'd110',
#                  'd101', 'd020', 'd011', 'd002', 'f300',
#                  'f210', 'f201', 'f120', 'f111', 'f102',
#                  'f030', 'f021', 'f012', 'f003']),
#    ("SolidHarmonic", [str(i) for i in range(8)])])
#
#def gui_base_widgets():
#    """New widgets for basic GUI functionality."""
#    return OrderedDict(scn_close=Button(icon="trash",
#                                 description=" Close",
#                                 layout=gui_lo),
#                       scn_clear=Button(icon="bomb",
#                                 description=" Clear",
#                                 layout=gui_lo),
#                       scn_saves=Button(icon="camera",
#                                 description=" Save",
#                                 layout=gui_lo))
#
#def gui_field_widgets(uni=False, test=False):
#    """New widgets for field GUI functionality."""
#    field_lims = {"min": 30, "max": 60, "value": 30,
#                  "step": 1, "layout": gui_lo,
#                  "continuous_update": False}
#    iso_lims = {"continuous_update": False,
#                "description": "Iso.",
#                "layout": gui_lo}
#    if uni: iso_lims.update({"min": 0.0001, "max": 0.1,
#                             "value": 0.0005, "step": 0.0005,
#                             "readout_format": ".4f"})
#    else: iso_lims.update({"min": 3.0, "max": 10.0, "value": 2.0})
#    if uni and not test: iso_lims["value"] = 0.03
#    return OrderedDict(field_iso=FloatSlider(**iso_lims),
#                       field_nx=IntSlider(description="Nx", **field_lims),
#                       field_ny=IntSlider(description="Ny", **field_lims),
#                       field_nz=IntSlider(description="Nz", **field_lims))
#
#
#################
## Base classes #
#################
#
=======
# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Universe Notebook Widget
#########################
"""

###
### OLDER CODE KEPT HERE FOR TESTING PURPOSES
###   AND TO COMPARE TIMINGS WITH VARIOUS
###   MESSAGE PASSING CONVENTIONS BETWEEN
###         PYTHON AND JAVASCRIPT
###      ** NOT IN PRODUCTION USE **
###
import os
import numpy as np
import pandas as pd
from glob import glob
from base64 import b64decode
from collections import OrderedDict
from traitlets import Bool, Float, Int, Instance, Unicode, List, Dict, Any, link
from ipywidgets import (Widget, DOMWidget, Box, widget_serialization, Layout,
                        Button, Dropdown, VBox, HBox, FloatSlider, IntSlider,
                        register, Play, jslink, Checkbox)
## Imports expected to break
from exatomic import Universe
from exatomic.base import sym2radius, sym2color


###################
# Default layouts #
###################

width = "400"
height = "400"
gui_lo = Layout(width="195px")


#######################
# Common GUI patterns #
#######################

uni_field_lists = OrderedDict([
    ("Hydrogenic", ['1s',   '2s',   '2px', '2py', '2pz',
                    '3s',   '3px',  '3py', '3pz',
                    '3d-2', '3d-1', '3d0', '3d+1', '3d+2']),
    ("Gaussian", ['s', 'px', 'py', 'pz', 'd200', 'd110',
                  'd101', 'd020', 'd011', 'd002', 'f300',
                  'f210', 'f201', 'f120', 'f111', 'f102',
                  'f030', 'f021', 'f012', 'f003']),
    ("SolidHarmonic", [str(i) for i in range(8)])])

def gui_base_widgets():
    """New widgets for basic GUI functionality."""
    return OrderedDict(scn_close=Button(icon="trash",
                                 description=" Close",
                                 layout=gui_lo),
                       scn_clear=Button(icon="bomb",
                                 description=" Clear",
                                 layout=gui_lo),
                       scn_saves=Button(icon="camera",
                                 description=" Save",
                                 layout=gui_lo))

def gui_field_widgets(uni=False, test=False):
    """New widgets for field GUI functionality."""
    field_lims = {"min": 30, "max": 60, "value": 30,
                  "step": 1, "layout": gui_lo,
                  "continuous_update": False}
    iso_lims = {"continuous_update": False,
                "description": "Iso.",
                "layout": gui_lo}
    if uni: iso_lims.update({"min": 0.0001, "max": 0.1,
                             "value": 0.0005, "step": 0.0005,
                             "readout_format": ".4f"})
    else: iso_lims.update({"min": 3.0, "max": 10.0, "value": 2.0})
    if uni and not test: iso_lims["value"] = 0.03
    return OrderedDict(field_iso=FloatSlider(**iso_lims),
                       field_nx=IntSlider(description="Nx", **field_lims),
                       field_ny=IntSlider(description="Ny", **field_lims),
                       field_nz=IntSlider(description="Nz", **field_lims))


################
# Base classes #
################

@register
class ExatomicScene(DOMWidget):
    """Resizable three.js scene."""
    _view_module = Unicode("jupyter-exatomic").tag(sync=True)
    _model_module = Unicode("jupyter-exatomic").tag(sync=True)
    _model_name = Unicode("ExatomicSceneModel").tag(sync=True)
    _view_name = Unicode("ExatomicSceneView").tag(sync=True)
    scn_clear = Bool(False).tag(sync=True)
    scn_saves = Bool(False).tag(sync=True)
    field_iso = Float(2.0).tag(sync=True)
    field_ox = Float(-3.0).tag(sync=True)
    field_oy = Float(-3.0).tag(sync=True)
    field_oz = Float(-3.0).tag(sync=True)
    field_fx = Float(3.0).tag(sync=True)
    field_fy = Float(3.0).tag(sync=True)
    field_fz = Float(3.0).tag(sync=True)
    field_nx = Int(31).tag(sync=True)
    field_ny = Int(31).tag(sync=True)
    field_nz = Int(31).tag(sync=True)
    savedir = Unicode().tag(sync=True)
    imgname = Unicode().tag(sync=True)

    def _handle_custom_msg(self, message, callback):
        """Custom message handler."""
        typ = message["type"]
        content = message["content"]
        if typ == "image": self._handle_image(content)

    def _handle_image(self, content):
        """Save a PNG of the scene."""
        savedir = self.savedir
        if not savedir: savedir = os.getcwd()
        fname = self.imgname
        if not fname:
            nxt = 0
            fname = "{:06d}.png".format(nxt)
            while os.path.isfile(os.sep.join([savedir, fname])):
                nxt += 1
                fname = "{:06d}.png".format(nxt)
        with open(os.sep.join([savedir, fname]), "wb") as f:
            f.write(b64decode(content.replace("data:image/png;base64,", "")))

    def __init__(self, *args, **kwargs):
        super(DOMWidget, self).__init__(*args,
                                        layout=Layout(width=width, height=height),
                                        **kwargs)

@register
class ExatomicBox(Box):
    """Base class for containers of a GUI and scene."""
    _model_module = Unicode("jupyter-exatomic").tag(sync=True)
    _view_module = Unicode("jupyter-exatomic").tag(sync=True)
    _model_name = Unicode("ExatomicBoxModel").tag(sync=True)
    _view_name = Unicode("ExatomicBoxView").tag(sync=True)
    scene = Instance(ExatomicScene, allow_none=True
                    ).tag(sync=True, **widget_serialization)

    def _close(self, b):
        """Shut down all active widgets within the container."""
        self.scene.close()
        for wid in self.controls:
            self.controls[wid].close()
        for wid in self.field_gui:
            self.field_gui[wid].close()
        if hasattr(self, 'uni_gui'):
            for wid in self.uni_gui:
                if isinstance(self.uni_gui[wid], dict):
                    for sub in self.uni_gui[wid]:
                        self.uni_gui[wid][sub].close()
                else: self.uni_gui[wid].close()
        self.close()

    def add_field_gui(self, key=None):
        """Add field controllers to the GUI widgets."""
        if key is not None:
            self.controls[key] = self.field_gui[key]
        else:
            for key, wid in self.field_gui.items():
                self.controls[key] = wid

    def remove_field_gui(self):
        """Remove field controllers from the GUI widgets."""
        for key in self.field_gui.keys():
            try: self.controls.pop(key)
            except KeyError: continue

    def init_gui(self, uni=False, test=False):
        """Initialize generic GUI controls and register callbacks."""
        self.controls = gui_base_widgets()
        def _scn_clear(b):
            self.scene.scn_clear = self.scene.scn_clear == False
        def _scn_saves(b):
            self.scene.scn_saves = self.scene.scn_saves == False
        self.controls['scn_close'].on_click(self._close)
        self.controls['scn_clear'].on_click(_scn_clear)
        self.controls['scn_saves'].on_click(_scn_saves)

        self.field_gui = gui_field_widgets(uni, test)
        def _field_iso(c): self.scene.field_iso = c.new
        def _field_nx(c): self.scene.field_nx = c.new
        def _field_ny(c): self.scene.field_ny = c.new
        def _field_nz(c): self.scene.field_nz = c.new
        self.field_gui['field_iso'].observe(_field_iso, names="value")
        self.field_gui['field_nx'].observe(_field_nx, names="value")
        self.field_gui['field_ny'].observe(_field_ny, names="value")
        self.field_gui['field_nz'].observe(_field_nz, names="value")

        if test and uni:
            self.uni_gui = {key: Dropdown(options=val, layout=gui_lo)
                            for key, val in uni_field_lists.items()}
            def _field_kind(c):
                self.scene.field_kind = c.new
                if 'field_ml' in self.controls:
                    self.controls.update([('field_ml',
                                           self.uni_gui['ml'][c.new])])
                self.gui = VBox([val for key, val in self.controls.items()],
                                 layout=Layout(width="200px"))
                self.set_gui()
            for wid in self.uni_gui:
                self.uni_gui[wid].observe(_field_kind, names="value")
            self.uni_gui['ml'] = {str(l): Dropdown(options=range(-l, l+1),
                                  layout=gui_lo) for l in range(8)}
            def _field_ml(d): self.scene.field_ml = d.new
            for wid in self.uni_gui['ml']:
                self.uni_gui['ml'][wid].observe(_field_ml, names="value")

    def set_gui(self):
        """Reset the GUI widgets."""
        self.children[0].children[0].children = self.gui.children
        self.on_displayed(Box._fire_children_displayed)

    def __init__(self, *args, scene=None, **kwargs):
        self.scene = ExatomicScene() if scene is None else scene
        if not hasattr(self, 'gui'): self.init_gui()
        self.children = [HBox([self.gui, self.scene])]
        super(ExatomicBox, self).__init__(*args,
                                            children=self.children,
                                            scene=self.scene,
                                            **kwargs)


@register
class ExatomicWidget(Widget):
    """Base widget for dataframe widgets."""
    _view_module = Unicode("jupyter-exatomic").tag(sync=True)
    _model_module = Unicode("jupyter-exatomic").tag(sync=True)
    _model_name = Unicode("ExatomicWidgetModel").tag(sync=True)
    _view_name = Unicode("ExatomicWidgetView").tag(sync=True)
    field = Int().tag(sync=True)

    def from_dataframe(self): return {}

    def __init__(self, df, *args, **kwargs):
        self.df = df
        dfargs = self.from_dataframe()
        kwargs.update(dfargs)
        super(ExatomicWidget, self).__init__(*args, **kwargs)


@register
class FieldWidget(ExatomicWidget):
    """Update field data in place."""
    _model_name = Unicode("FieldWidgetModel").tag(sync=True)
    _view_name = Unicode("FieldWidgetView").tag(sync=True)
    indices = List().tag(sync=True)
    values = List().tag(sync=True)
    params = Dict().tag(sync=True)

    def update_traits(self, fdx, fldx):
        grp = self.df.groupby('frame').get_group(fdx)
        self.indices = grp.index.values.tolist()
        idx = self.indices[fldx]
        self.params = grp.loc[idx].to_dict()
        self.values = self.df.field_values[idx].tolist()

    def from_dataframe(self):
        self.df['frame'] = self.df['frame'].astype(int)
        self.df['nx'] = self.df['nx'].astype(int)
        self.df['ny'] = self.df['ny'].astype(int)
        self.df['nz'] = self.df['nz'].astype(int)
        if not all((col in self.df.columns for col in ['fx', 'fy', 'fz'])):
            for d, l in [('x', 'i'), ('y', 'j'), ('z', 'k')]:
                self.df['f'+d] = self.df['o'+d] + (self.df['n'+d] - 1) * self.df['d'+d+l]
        grp = self.df.groupby('frame').get_group(0)
        return {'values': self.df.field_values[0].tolist(),
                'indices': grp.index.values.tolist(),
                'params': grp.loc[grp.index.values[0]].to_dict()}




@register
class TwoWidget(ExatomicWidget):
    """Update two data in place."""
    _model_name = Unicode("TwoWidgetModel").tag(sync=True)
    _view_name = Unicode("TwoWidgetView").tag(sync=True)
    b0 = List().tag(sync=True)
    b1 = List().tag(sync=True)

    def update_traits(self, idx):
        grp = self.df.groupby('frame').get_group(idx)
        bonded = grp.loc[self.df['bond'] == True]
        self.b0 = bonded['atom0'].tolist()
        self.b1 = bonded['atom1'].tolist()

    def from_dataframe(self, lbls):
        grp = self.df.groupby('frame').get_group(0)
        bonded = grp.loc[self.df['bond'] == True]
        return {'b0': bonded['atom0'].tolist(),
                'b1': bonded['atom1'].tolist()}

    def __init__(self, df, lbls, *args, **kwargs):
        self.df = df
        dfargs = self.from_dataframe(lbls)
        kwargs.update(dfargs)
        super(ExatomicWidget, self).__init__(*args, **kwargs)


@register
class AtomWidget(ExatomicWidget):
    """Update atom data in place."""
    _model_name = Unicode("AtomWidgetModel").tag(sync=True)
    _view_name = Unicode("AtomWidgetView").tag(sync=True)
    x = List().tag(sync=True)
    y = List().tag(sync=True)
    z = List().tag(sync=True)
    s = List().tag(sync=True)
    r = Dict().tag(sync=True)
    c = Dict().tag(sync=True)

    def update_traits(self, idx):
        grp = self.df.groupby('frame').get_group(idx)
        self.x = grp['x'].tolist()
        self.y = grp['y'].tolist()
        self.z = grp['z'].tolist()
        self.s = grp['symbol'].cat.codes.values.tolist()

    def from_dataframe(self, precision=2):
        traits = {}
        grp = self.df.groupby('frame').get_group(0)
        traits['x'] = grp['x'].tolist()
        traits['y'] = grp['y'].tolist()
        traits['z'] = grp['z'].tolist()
        symmap = {i: v for i, v in enumerate(grp['symbol'].cat.categories)
                  if v in self.df.unique_atoms}
        unq = self.df['symbol'].unique()
        radii = sym2radius[unq]
        colors = sym2color[unq]
        traits['s'] =  grp['symbol'].cat.codes.values.tolist()
        traits['r'] = {i: 0.5 * radii[v] for i, v in symmap.items()}
        traits['c'] = {i: colors[v] for i, v in symmap.items()}
        return traits



@register
class SmallverseScene(ExatomicScene):
    _model_name = Unicode("SmallverseSceneModel").tag(sync=True)
    _view_name = Unicode("SmallverseSceneView").tag(sync=True)
    atom = Instance(AtomWidget, allow_none=True).tag(sync=True, **widget_serialization)
    atom_spheres = Bool(False).tag(sync=True)
    field = Instance(FieldWidget, allow_none=True).tag(sync=True, **widget_serialization)
    frame = Instance(FrameWidget, allow_none=True).tag(sync=True, **widget_serialization)
    two = Instance(TwoWidget, allow_none=True).tag(sync=True, **widget_serialization)
    frame_idx = Int(0).tag(sync=True)
    field_idx = Any().tag(sync=True)
    field_iso = Float(0.03).tag(sync=True)

@register
class SmallverseWidget(ExatomicBox):
    _model_name = Unicode("SmallverseWidgetModel").tag(sync=True)
    _view_name = Unicode("SmallverseWidgetView").tag(sync=True)
    field_show = Bool(False).tag(sync=True)

    def init_gui(self, nframes=1, fields=None):
        super(SmallverseWidget, self).init_gui(uni=True, test=False)
        playable = bool(nframes <= 1)
        frame_lims = {'min': 0, 'max': nframes-1, 'step': 1,
                      'value': 0, 'layout': gui_lo}
        self.controls.update([('scn_frame', IntSlider(
                               description='Frame', **frame_lims)),
                              ('playing', Play(description="Press play",
                               disabled=playable, **frame_lims)),
                              ('atom_spheres', Button(description=" Atoms",
                               icon="adjust", layout=gui_lo))])
        if fields is not None:
            print('fields is not None:', fields)
            self.controls.update([('field_show', Button(description=" Fields",
                                  layout=gui_lo, icon="cube"))])
            def _field_show(b):
                self.field_show = self.field_show == False
                fdx = self.scene.field_idx
                if self.field_show:
                    self.controls.update([('field_options', Dropdown(options=fields,
                                          layout=gui_lo))])
                    self.add_field_gui('field_iso')
                    def _field_options(c):
                        self.scene.field_idx = c.new
                        self.scene.field.update_traits(self.scene.frame_idx, c.new)
                    self.controls['field_options'].observe(_field_options, names="value")
                else:
                    self.remove_field_gui()
                    self.controls.pop('field_options')

                self.gui = VBox([val for key, val in self.controls.items()],
                                 layout=Layout(width="200px"))
                self.set_gui()
            self.controls['field_show'].on_click(_field_show)
        def _scn_frame(c):
            self.scene.frame_idx = c.new
            self.scene.atom.update_traits(c.new)
            self.scene.two.update_traits(c.new)
        def _atom_spheres(b):
            self.scene.atom_spheres = self.scene.atom_spheres == False
        self.controls['scn_frame'].observe(_scn_frame, names='value')
        self.controls['atom_spheres'].on_click(_atom_spheres)
        jslink((self.controls['playing'], 'value'), (self.controls['scn_frame'], 'value'))
        self.gui = VBox([val for key, val in self.controls.items()],
                            layout=Layout(width="200px"))



    def __init__(self, uni, *args, **kwargs):
        if not isinstance(uni, Universe):
            raise TypeError("Object passed to UniverseWidget must be a universe.")
        scene = SmallverseScene()
        try: scene.atom = AtomWidget(uni.atom)
        except AttributeError: scene.atom = None
        try: scene.frame = FrameWidget(uni.frame)
        except AttributeError: scene.frame = None
        try:
            scene.field = FieldWidget(uni.field)
            fields = ['null'] + scene.field.indices
        except AttributeError: scene.field, fields = None, None
        try: scene.two = TwoWidget(uni.atom_two, uni.get_atom_labels())
        except AttributeError: scene.two = None
        self.init_gui(nframes=uni.atom.nframes, fields=fields)
        super(SmallverseWidget, self).__init__(*args,
                                             scene=scene,
                                             **kwargs)

>>>>>>> 1c37655b6be3dca60b2adbeee8ca3767e5477943:exatomic/_widget.py
#@register
#class ExatomicScene(DOMWidget):
#    """Resizable three.js scene."""
#    _view_module = Unicode("jupyter-exatomic").tag(sync=True)
#    _model_module = Unicode("jupyter-exatomic").tag(sync=True)
#    _model_name = Unicode("ExatomicSceneModel").tag(sync=True)
#    _view_name = Unicode("ExatomicSceneView").tag(sync=True)
#    scn_clear = Bool(False).tag(sync=True)
#    scn_saves = Bool(False).tag(sync=True)
#    field_iso = Float(2.0).tag(sync=True)
#    field_ox = Float(-3.0).tag(sync=True)
#    field_oy = Float(-3.0).tag(sync=True)
#    field_oz = Float(-3.0).tag(sync=True)
#    field_fx = Float(3.0).tag(sync=True)
#    field_fy = Float(3.0).tag(sync=True)
#    field_fz = Float(3.0).tag(sync=True)
#    field_nx = Int(31).tag(sync=True)
#    field_ny = Int(31).tag(sync=True)
#    field_nz = Int(31).tag(sync=True)
#    savedir = Unicode().tag(sync=True)
#    imgname = Unicode().tag(sync=True)
#
#    def _handle_custom_msg(self, message, callback):
#        """Custom message handler."""
#        typ = message["type"]
#        content = message["content"]
#        if typ == "image": self._handle_image(content)
#
#    def _handle_image(self, content):
#        """Save a PNG of the scene."""
#        savedir = self.savedir
#        if not savedir: savedir = os.getcwd()
#        fname = self.imgname
#        if not fname:
#            nxt = 0
#            fname = "{:06d}.png".format(nxt)
#            while os.path.isfile(os.sep.join([savedir, fname])):
#                nxt += 1
#                fname = "{:06d}.png".format(nxt)
#        with open(os.sep.join([savedir, fname]), "wb") as f:
#            f.write(b64decode(content.replace("data:image/png;base64,", "")))
#
#    def __init__(self, *args, **kwargs):
#        super(DOMWidget, self).__init__(*args,
#                                        layout=Layout(width=width, height=height),
#                                        **kwargs)
#
#@register
#class ExatomicBox(Box):
#    """Base class for containers of a GUI and scene."""
#    _model_module = Unicode("jupyter-exatomic").tag(sync=True)
#    _view_module = Unicode("jupyter-exatomic").tag(sync=True)
#    _model_name = Unicode("ExatomicBoxModel").tag(sync=True)
#    _view_name = Unicode("ExatomicBoxView").tag(sync=True)
#    scene = Instance(ExatomicScene, allow_none=True
#                    ).tag(sync=True, **widget_serialization)
#
#    def _close(self, b):
#        """Shut down all active widgets within the container."""
#        self.scene.close()
#        for wid in self.controls:
#            self.controls[wid].close()
#        for wid in self.field_gui:
#            self.field_gui[wid].close()
#        if hasattr(self, 'uni_gui'):
#            for wid in self.uni_gui:
#                if isinstance(self.uni_gui[wid], dict):
#                    for sub in self.uni_gui[wid]:
#                        self.uni_gui[wid][sub].close()
#                else: self.uni_gui[wid].close()
#        self.close()
#
#    def add_field_gui(self, key=None):
#        """Add field controllers to the GUI widgets."""
#        if key is not None:
#            self.controls[key] = self.field_gui[key]
#        else:
#            for key, wid in self.field_gui.items():
#                self.controls[key] = wid
#
#    def remove_field_gui(self):
#        """Remove field controllers from the GUI widgets."""
#        for key in self.field_gui.keys():
#            try: self.controls.pop(key)
#            except KeyError: continue
#
#    def init_gui(self, uni=False, test=False):
#        """Initialize generic GUI controls and register callbacks."""
#        self.controls = gui_base_widgets()
#        def _scn_clear(b):
#            self.scene.scn_clear = self.scene.scn_clear == False
#        def _scn_saves(b):
#            self.scene.scn_saves = self.scene.scn_saves == False
#        self.controls['scn_close'].on_click(self._close)
#        self.controls['scn_clear'].on_click(_scn_clear)
#        self.controls['scn_saves'].on_click(_scn_saves)
#
#        self.field_gui = gui_field_widgets(uni, test)
#        def _field_iso(c): self.scene.field_iso = c.new
#        def _field_nx(c): self.scene.field_nx = c.new
#        def _field_ny(c): self.scene.field_ny = c.new
#        def _field_nz(c): self.scene.field_nz = c.new
#        self.field_gui['field_iso'].observe(_field_iso, names="value")
#        self.field_gui['field_nx'].observe(_field_nx, names="value")
#        self.field_gui['field_ny'].observe(_field_ny, names="value")
#        self.field_gui['field_nz'].observe(_field_nz, names="value")
#
#        if test and uni:
#            self.uni_gui = {key: Dropdown(options=val, layout=gui_lo)
#                            for key, val in uni_field_lists.items()}
#            def _field_kind(c):
#                self.scene.field_kind = c.new
#                if 'field_ml' in self.controls:
#                    self.controls.update([('field_ml',
#                                           self.uni_gui['ml'][c.new])])
#                self.gui = VBox([val for key, val in self.controls.items()],
#                                 layout=Layout(width="200px"))
#                self.set_gui()
#            for wid in self.uni_gui:
#                self.uni_gui[wid].observe(_field_kind, names="value")
#            self.uni_gui['ml'] = {str(l): Dropdown(options=range(-l, l+1),
#                                  layout=gui_lo) for l in range(8)}
#            def _field_ml(d): self.scene.field_ml = d.new
#            for wid in self.uni_gui['ml']:
#                self.uni_gui['ml'][wid].observe(_field_ml, names="value")
#
#    def set_gui(self):
#        """Reset the GUI widgets."""
#        self.children[0].children[0].children = self.gui.children
#        self.on_displayed(Box._fire_children_displayed)
#
#    def __init__(self, *args, scene=None, **kwargs):
#        self.scene = ExatomicScene() if scene is None else scene
#        if not hasattr(self, 'gui'): self.init_gui()
#        self.children = [HBox([self.gui, self.scene])]
#        super(ExatomicBox, self).__init__(*args,
#                                            children=self.children,
#                                            scene=self.scene,
#                                            **kwargs)
#
#
#@register
#class ExatomicWidget(Widget):
#    """Base widget for dataframe widgets."""
#    _view_module = Unicode("jupyter-exatomic").tag(sync=True)
#    _model_module = Unicode("jupyter-exatomic").tag(sync=True)
#    _model_name = Unicode("ExatomicWidgetModel").tag(sync=True)
#    _view_name = Unicode("ExatomicWidgetView").tag(sync=True)
#    field = Int().tag(sync=True)
#
#    def from_dataframe(self): return {}
#
#    def __init__(self, df, *args, **kwargs):
#        self.df = df
#        dfargs = self.from_dataframe()
#        kwargs.update(dfargs)
#        super(ExatomicWidget, self).__init__(*args, **kwargs)
#
#
#@register
#class FieldWidget(ExatomicWidget):
#    """Update field data in place."""
#    _model_name = Unicode("FieldWidgetModel").tag(sync=True)
#    _view_name = Unicode("FieldWidgetView").tag(sync=True)
#    indices = List().tag(sync=True)
#    values = List().tag(sync=True)
#    params = Dict().tag(sync=True)
#
#    def update_traits(self, fdx, fldx):
#        grp = self.df.groupby('frame').get_group(fdx)
#        self.indices = grp.index.values.tolist()
#        idx = self.indices[fldx]
#        self.params = grp.loc[idx].to_dict()
#        self.values = self.df.field_values[idx].tolist()
#
#    def from_dataframe(self):
#        self.df['frame'] = self.df['frame'].astype(int)
#        self.df['nx'] = self.df['nx'].astype(int)
#        self.df['ny'] = self.df['ny'].astype(int)
#        self.df['nz'] = self.df['nz'].astype(int)
#        if not all((col in self.df.columns for col in ['fx', 'fy', 'fz'])):
#            for d, l in [('x', 'i'), ('y', 'j'), ('z', 'k')]:
#                self.df['f'+d] = self.df['o'+d] + (self.df['n'+d] - 1) * self.df['d'+d+l]
#        grp = self.df.groupby('frame').get_group(0)
#        return {'values': self.df.field_values[0].tolist(),
#                'indices': grp.index.values.tolist(),
#                'params': grp.loc[grp.index.values[0]].to_dict()}
#
#
#
#
#@register
<<<<<<< HEAD:exatomic/widget_small.py
#class TwoWidget(ExatomicWidget):
#    """Update two data in place."""
#    _model_name = Unicode("TwoWidgetModel").tag(sync=True)
#    _view_name = Unicode("TwoWidgetView").tag(sync=True)
#    b0 = List().tag(sync=True)
#    b1 = List().tag(sync=True)
#
#    def update_traits(self, idx):
#        grp = self.df.groupby('frame').get_group(idx)
#        bonded = grp.loc[self.df['bond'] == True]
#        self.b0 = bonded['atom0'].tolist()
#        self.b1 = bonded['atom1'].tolist()
#
#    def from_dataframe(self, lbls):
#        grp = self.df.groupby('frame').get_group(0)
#        bonded = grp.loc[self.df['bond'] == True]
#        return {'b0': bonded['atom0'].tolist(),
#                'b1': bonded['atom1'].tolist()}
=======
#class AllTwoWidget(ExatomicWidget):
#    """Pass all two data to javascript once."""
#    _model_name = Unicode("AllTwoWidgetModel").tag(sync=True)
#    _view_name = Unicode("AllTwoWidgetView").tag(sync=True)
#    b0 = Unicode().tag(sync=True)
#    b1 = Unicode().tag(sync=True)
#
#    def from_dataframe(self, lbls):
#        bonded = self.df.ix[self.df['bond'] == True, ['atom0', 'atom1', 'frame']]
#        lbl0 = bonded['atom0'].map(lbls)
#        lbl1 = bonded['atom1'].map(lbls)
#        lbl = pd.concat((lbl0, lbl1), axis=1)
#        lbl['frame'] = bonded['frame']
#        bond_grps = lbl.groupby('frame')
#        frames = self.df['frame'].unique().astype(np.int64)
#        b0 = np.empty((len(frames), ), dtype='O')
#        b1 = b0.copy()
#        for i, frame in enumerate(frames):
#            try:
#                b0[i] = bond_grps.get_group(frame)['atom0'].astype(np.int64).values
#                b1[i] = bond_grps.get_group(frame)['atom1'].astype(np.int64).values
#            except Exception:
#                b0[i] = []
#                b1[i] = []
#        b0 = pd.Series(b0).to_json(orient='values')
#        b1 = pd.Series(b1).to_json(orient='values')
#        return {'b0': b0, 'b1': b1}
>>>>>>> 1c37655b6be3dca60b2adbeee8ca3767e5477943:exatomic/_widget.py
#
#    def __init__(self, df, lbls, *args, **kwargs):
#        self.df = df
#        dfargs = self.from_dataframe(lbls)
#        kwargs.update(dfargs)
#        super(ExatomicWidget, self).__init__(*args, **kwargs)
#
#
#@register
#class AtomWidget(ExatomicWidget):
#    """Update atom data in place."""
#    _model_name = Unicode("AtomWidgetModel").tag(sync=True)
#    _view_name = Unicode("AtomWidgetView").tag(sync=True)
#    x = List().tag(sync=True)
#    y = List().tag(sync=True)
#    z = List().tag(sync=True)
#    s = List().tag(sync=True)
#    r = Dict().tag(sync=True)
#    c = Dict().tag(sync=True)
#
#    def update_traits(self, idx):
#        grp = self.df.groupby('frame').get_group(idx)
#        self.x = grp['x'].tolist()
#        self.y = grp['y'].tolist()
#        self.z = grp['z'].tolist()
#        self.s = grp['symbol'].cat.codes.values.tolist()
#
#    def from_dataframe(self, precision=2):
#        traits = {}
<<<<<<< HEAD:exatomic/widget_small.py
#        grp = self.df.groupby('frame').get_group(0)
#        traits['x'] = grp['x'].tolist()
#        traits['y'] = grp['y'].tolist()
#        traits['z'] = grp['z'].tolist()
#        symmap = {i: v for i, v in enumerate(grp['symbol'].cat.categories)
=======
#        grps = self.df.groupby('frame')
#        for col in ['x', 'y', 'z']:
#            traits[col] = grps.apply(
#                lambda y: y[col].to_json(
#                orient='values', double_precision=3)
#                ).to_json(orient="values").replace('"', '')
#        grps = self.df.groupby('frame')
#        syms = grps.apply(lambda g: g['symbol'].cat.codes.values)
#        symmap = {i: v for i, v in enumerate(self.df['symbol'].cat.categories)
>>>>>>> 1c37655b6be3dca60b2adbeee8ca3767e5477943:exatomic/_widget.py
#                  if v in self.df.unique_atoms}
#        unq = self.df['symbol'].unique()
#        radii = symbol_to_radius()[unq]
#        colors = symbol_to_color()[unq]
#        traits['s'] =  grp['symbol'].cat.codes.values.tolist()
#        traits['r'] = {i: 0.5 * radii[v] for i, v in symmap.items()}
#        traits['c'] = {i: colors[v] for i, v in symmap.items()}
#        return traits
<<<<<<< HEAD:exatomic/widget_small.py
#
#
#
#@register
#class SmallverseScene(ExatomicScene):
#    _model_name = Unicode("SmallverseSceneModel").tag(sync=True)
#    _view_name = Unicode("SmallverseSceneView").tag(sync=True)
#    atom = Instance(AtomWidget, allow_none=True).tag(sync=True, **widget_serialization)
#    atom_spheres = Bool(False).tag(sync=True)
#    field = Instance(FieldWidget, allow_none=True).tag(sync=True, **widget_serialization)
#    frame = Instance(FrameWidget, allow_none=True).tag(sync=True, **widget_serialization)
#    two = Instance(TwoWidget, allow_none=True).tag(sync=True, **widget_serialization)
#    frame_idx = Int(0).tag(sync=True)
#    field_idx = Any().tag(sync=True)
#    field_iso = Float(0.03).tag(sync=True)
#
=======




>>>>>>> 1c37655b6be3dca60b2adbeee8ca3767e5477943:exatomic/_widget.py
#@register
#class SmallverseWidget(ExatomicBox):
#    _model_name = Unicode("SmallverseWidgetModel").tag(sync=True)
#    _view_name = Unicode("SmallverseWidgetView").tag(sync=True)
#    field_show = Bool(False).tag(sync=True)
#
#    def init_gui(self, nframes=1, fields=None):
#        super(SmallverseWidget, self).init_gui(uni=True, test=False)
#        playable = bool(nframes <= 1)
#        frame_lims = {'min': 0, 'max': nframes-1, 'step': 1,
#                      'value': 0, 'layout': gui_lo}
#        self.controls.update([('scn_frame', IntSlider(
#                               description='Frame', **frame_lims)),
#                              ('playing', Play(description="Press play",
#                               disabled=playable, **frame_lims)),
#                              ('atom_spheres', Button(description=" Atoms",
#                               icon="adjust", layout=gui_lo))])
#        if fields is not None:
#            print('fields is not None:', fields)
#            self.controls.update([('field_show', Button(description=" Fields",
#                                  layout=gui_lo, icon="cube"))])
#            def _field_show(b):
#                self.field_show = self.field_show == False
#                fdx = self.scene.field_idx
#                if self.field_show:
#                    self.controls.update([('field_options', Dropdown(options=fields,
#                                          layout=gui_lo))])
#                    self.add_field_gui('field_iso')
#                    def _field_options(c):
#                        self.scene.field_idx = c.new
#                        self.scene.field.update_traits(self.scene.frame_idx, c.new)
#                    self.controls['field_options'].observe(_field_options, names="value")
#                else:
#                    self.remove_field_gui()
#                    self.controls.pop('field_options')
#
#                self.gui = VBox([val for key, val in self.controls.items()],
#                                 layout=Layout(width="200px"))
#                self.set_gui()
#            self.controls['field_show'].on_click(_field_show)
#        def _scn_frame(c):
#            self.scene.frame_idx = c.new
#            self.scene.atom.update_traits(c.new)
#            self.scene.two.update_traits(c.new)
#        def _atom_spheres(b):
#            self.scene.atom_spheres = self.scene.atom_spheres == False
#        self.controls['scn_frame'].observe(_scn_frame, names='value')
#        self.controls['atom_spheres'].on_click(_atom_spheres)
#        jslink((self.controls['playing'], 'value'), (self.controls['scn_frame'], 'value'))
#        self.gui = VBox([val for key, val in self.controls.items()],
#                            layout=Layout(width="200px"))
#
#
#
#    def __init__(self, uni, *args, **kwargs):
#        if not isinstance(uni, Universe):
#            raise TypeError("Object passed to UniverseWidget must be a universe.")
#        scene = SmallverseScene()
#        try: scene.atom = AtomWidget(uni.atom)
#        except AttributeError: scene.atom = None
#        try: scene.frame = FrameWidget(uni.frame)
#        except AttributeError: scene.frame = None
#        try:
#            scene.field = FieldWidget(uni.field)
#            fields = ['null'] + scene.field.indices
#        except AttributeError: scene.field, fields = None, None
#        try: scene.two = TwoWidget(uni.atom_two, uni.get_atom_labels())
#        except AttributeError: scene.two = None
#        self.init_gui(nframes=uni.atom.nframes, fields=fields)
#        super(SmallverseWidget, self).__init__(*args,
#                                             scene=scene,
#                                             **kwargs)
#
##@register
##class AllFieldWidget(ExatomicWidget):
##    """Pass all field data to javascript once."""
##    _model_name = Unicode("AllFieldWidgetModel").tag(sync=True)
##    _view_name = Unicode("AllFieldWidgetView").tag(sync=True)
##    indices = List().tag(sync=True)
##    values = List().tag(sync=True)
##    params = Dict().tag(sync=True)
##
##    def from_dataframe(self):
##        self.df['frame'] = self.df['frame'].astype(int)
##        self.df['nx'] = self.df['nx'].astype(int)
##        self.df['ny'] = self.df['ny'].astype(int)
##        self.df['nz'] = self.df['nz'].astype(int)
##        if not all((col in self.df.columns for col in ['fx', 'fy', 'fz'])):
##            for d, l in [('x', 'i'), ('y', 'j'), ('z', 'k')]:
##                self.df['f'+d] = self.df['o'+d] + (self.df['n'+d] - 1) * self.df['d'+d+l]
##        grps = self.df.groupby('frame')
##        fps = grps.apply(lambda x: x[['ox', 'oy', 'oz',
##                                      'nx', 'ny', 'nz',
##                                      'fx', 'fy', 'fz']].T.to_dict()).to_dict()
##        try: idxs = list(map(list, grps.groups.values()))
##        except: idxs = [list(grp.index) for i, grp in grps]
##        vals = [f.tolist() for f in self.df.field_values]
##        return {'values': vals, 'indices': idxs, 'params': fps}
##
##@register
##class AllTwoWidget(ExatomicWidget):
##    """Pass all two data to javascript once."""
##    _model_name = Unicode("AllTwoWidgetModel").tag(sync=True)
##    _view_name = Unicode("AllTwoWidgetView").tag(sync=True)
##    b0 = Unicode().tag(sync=True)
##    b1 = Unicode().tag(sync=True)
##
##    def from_dataframe(self, lbls):
##        bonded = self.df.ix[self.df['bond'] == True, ['atom0', 'atom1', 'frame']]
##        lbl0 = bonded['atom0'].map(lbls)
##        lbl1 = bonded['atom1'].map(lbls)
##        lbl = pd.concat((lbl0, lbl1), axis=1)
##        lbl['frame'] = bonded['frame']
##        bond_grps = lbl.groupby('frame')
##        frames = self.df['frame'].unique().astype(np.int64)
##        b0 = np.empty((len(frames), ), dtype='O')
##        b1 = b0.copy()
##        for i, frame in enumerate(frames):
##            try:
##                b0[i] = bond_grps.get_group(frame)['atom0'].astype(np.int64).values
##                b1[i] = bond_grps.get_group(frame)['atom1'].astype(np.int64).values
##            except Exception:
##                b0[i] = []
##                b1[i] = []
##        b0 = pd.Series(b0).to_json(orient='values')
##        b1 = pd.Series(b1).to_json(orient='values')
##        return {'b0': b0, 'b1': b1}
##
##    def __init__(self, df, lbls, *args, **kwargs):
##        self.df = df
##        dfargs = self.from_dataframe(lbls)
##        kwargs.update(dfargs)
##        super(ExatomicWidget, self).__init__(*args, **kwargs)
##
##@register
##class AllAtomWidget(ExatomicWidget):
##    """Pass all atom data to javascript once."""
##    _model_name = Unicode("AllAtomWidgetModel").tag(sync=True)
##    _view_name = Unicode("AllAtomWidgetView").tag(sync=True)
##    x = Unicode().tag(sync=True)
##    y = Unicode().tag(sync=True)
##    z = Unicode().tag(sync=True)
##    syms = Unicode().tag(sync=True)
##    radii = Dict().tag(sync=True)
##    colrs = Dict().tag(sync=True)
##
##    def from_dataframe(self, precision=2):
##        traits = {}
##        grps = self.df.groupby('frame')
##        for col in ['x', 'y', 'z']:
##            traits[col] = grps.apply(
##                lambda y: y[col].to_json(
##                orient='values', double_precision=3)
##                ).to_json(orient="values").replace('"', '')
##        grps = self.df.groupby('frame')
##        syms = grps.apply(lambda g: g['symbol'].cat.codes.values)
##        symmap = {i: v for i, v in enumerate(self.df['symbol'].cat.categories)
##                  if v in self.df.unique_atoms}
##        unq = self.df['symbol'].unique()
##        radii = symbol_to_radius()[unq]
##        colors = symbol_to_color()[unq]
##        traits['syms'] = syms.to_json(orient='values')
##        traits['radii'] = {i: 0.5 * radii[v] for i, v in symmap.items()}
##        traits['colrs'] = {i: colors[v] for i, v in symmap.items()}
##        return traits
#
#
#
#
##@register
##class FrameWidget(ExatomicWidget):
##    """Pass frame data to javascript."""
##    _model_name = Unicode("FrameWidgetModel").tag(sync=True)
##    _view_name = Unicode("FrameWidgetView").tag(sync=True)
##    xi = Float().tag(sync=True)
##    xj = Float().tag(sync=True)
##    xk = Float().tag(sync=True)
##    yi = Float().tag(sync=True)
##    yj = Float().tag(sync=True)
##    yk = Float().tag(sync=True)
##    zi = Float().tag(sync=True)
##    zj = Float().tag(sync=True)
##    zk = Float().tag(sync=True)
##    ox = Float().tag(sync=True)
##    oy = Float().tag(sync=True)
##    oz = Float().tag(sync=True)
#
#
