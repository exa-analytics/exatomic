# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Universe Notebook Widget
#########################
"""
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
from exa.relational.isotope import symbol_to_radius, symbol_to_color

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
                    self.scene.field_ml = self.uni_gui['ml'][c.new].options[0]
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
        if not hasattr(self, 'gui'): self.gui = VBox([])
        self.children = [HBox([self.gui, self.scene])]
        super(ExatomicBox, self).__init__(*args,
                                            children=self.children,
                                            scene=self.scene,
                                            **kwargs)

########################
# Basic example widget #
########################

@register
class TestScene(ExatomicScene):
    """A basic scene to test some javascript."""
    _model_name = Unicode("TestSceneModel").tag(sync=True)
    _view_name = Unicode("TestSceneView").tag(sync=True)
    geo_shape = Bool(True).tag(sync=True)
    field = Unicode("null").tag(sync=True)

@register
class TestContainer(ExatomicBox):
    """A basic container to test some javascript."""
    _model_name = Unicode("TestContainerModel").tag(sync=True)
    _view_name = Unicode("TestContainerView").tag(sync=True)

    def init_gui(self):
        """Initialize specific GUI controls and register callbacks."""
        super(TestContainer, self).init_gui()
        fopts = ['null', 'sphere', 'torus', 'ellipsoid']
        self.controls.update([('geo_shape', Button(icon="cubes", 
                               description="  Mesh", layout=gui_lo)),
                              ('field_options', Dropdown(options=fopts,
                               layout=gui_lo))])
        def _geo_shape(b): 
            self.scene.geo_shape = self.scene.geo_shape == False
        self.controls['geo_shape'].on_click(_geo_shape)
        def _field(c):
            self.scene.field = c.new
            if c.new == 'null': self.remove_field_gui()
            else: self.add_field_gui()
            self.gui = VBox([val for key, val in self.controls.items()],
                             layout=Layout(width="200px"))
            self.set_gui()
        self.controls['field_options'].observe(_field, names="value")
        self.gui = VBox([val for key, val in self.controls.items()],
                            layout=Layout(width="200px"))

    def __init__(self, *args, **kwargs):
        super(TestContainer, self).__init__(*args,
                                            scene=TestScene(),
                                            **kwargs)


###########################
# Universe example widget #
###########################

@register
class TestUniverseScene(ExatomicScene):
    """Test :class:`~exatomic.container.Universe` scene."""
    _model_name = Unicode("TestUniverseSceneModel").tag(sync=True)
    _view_name = Unicode("TestUniverseSceneView").tag(sync=True)
    field_iso = Float(0.0005).tag(sync=True)
    field = Unicode('Hydrogenic').tag(sync=True)
    field_kind = Unicode('1s').tag(sync=True)
    field_ox = Float(-30.0).tag(sync=True)
    field_oy = Float(-30.0).tag(sync=True)
    field_oz = Float(-30.0).tag(sync=True)
    field_fx = Float(30.0).tag(sync=True)
    field_fy = Float(30.0).tag(sync=True)
    field_fz = Float(30.0).tag(sync=True)
    field_ml = Int(0).tag(sync=True)


@register
class TestUniverse(ExatomicBox):
    """Test :class:`~exatomic.container.Universe` test widget."""
    _model_name = Unicode("TestUniverseModel").tag(sync=True)
    _view_name = Unicode("TestUniverseView").tag(sync=True)

    def init_gui(self):
        super(TestUniverse, self).init_gui(uni=True, test=True)
        opts = uni_field_lists.keys()
        self.controls.update([('field_options', Dropdown(options=opts, 
                               layout=gui_lo)),
                              ('field_kind', self.uni_gui[self.scene.field])])
        self.controls.update(self.field_gui)
        def _field(c):
            self.scene.field = c.new
            fk = uni_field_lists[c.new][0]
            self.scene.field_kind = fk 
            if self.scene.field == 'SolidHarmonic':  
                self.remove_field_gui()
                self.controls.update([('field_ml', self.uni_gui['ml'][fk])])
                self.add_field_gui()
            elif 'field_ml' in self.controls: self.controls.pop('field_ml')
            self.controls['field_kind'] = self.uni_gui[c.new]
            self.gui = VBox([val for key, val in self.controls.items()],
                             layout=Layout(width="200px"))
            self.set_gui()
        self.controls['field_options'].observe(_field, names="value")
        self.gui = VBox([val for key, val in self.controls.items()],
                            layout=Layout(width="200px"))

    def __init__(self, *args, **kwargs):
        super(TestUniverse, self).__init__(*args,
                                            scene=TestUniverseScene(),
                                            **kwargs)


################################
# Universe and related widgets #
################################

def atom_traits(df):
    """Get atom table traitlets."""
    traits = {}
    grps = df.groupby('frame')
    for col in ['x', 'y', 'z']:
        traits['atom_' + col] = grps.apply(
            lambda y: y[col].to_json(
            orient='values', double_precision=3)
            ).to_json(orient="values").replace('"', '')
    grps = df.groupby('frame')
    syms = grps.apply(lambda g: g['symbol'].cat.codes.values)
    symmap = {i: v for i, v in enumerate(df['symbol'].cat.categories) 
              if v in df.unique_atoms}
    unq = df['symbol'].unique()
    radii = symbol_to_radius()[unq]
    colors = symbol_to_color()[unq]
    traits['atom_s'] = syms.to_json(orient='values')
    traits['atom_r'] = {i: 0.5 * radii[v] for i, v in symmap.items()}
    traits['atom_c'] = {i: colors[v] for i, v in symmap.items()}
    return traits

def field_traits(df):
    """Get field table traitlets."""
    df['frame'] = df['frame'].astype(int)
    df['nx'] = df['nx'].astype(int)
    df['ny'] = df['ny'].astype(int)
    df['nz'] = df['nz'].astype(int)
    if not all((col in df.columns for col in ['fx', 'fy', 'fz'])):
        for d, l in [('x', 'i'), ('y', 'j'), ('z', 'k')]:
            df['f'+d] = df['o'+d] + (df['n'+d] - 1) * df['d'+d+l]
    grps = df.groupby('frame')
    fps = grps.apply(lambda x: x[['ox', 'oy', 'oz',
                                  'nx', 'ny', 'nz',
                                  'fx', 'fy', 'fz']].T.to_dict()).to_dict()
    try: idxs = list(map(list, grps.groups.values()))
    except: idxs = [list(grp.index) for i, grp in grps]
    vals = [f.tolist() for f in df.field_values]
    return {'field_v': vals, 'field_i': idxs, 'field_p': fps}

def two_traits(df, lbls):
    """Get two table traitlets."""
    bonded = df.ix[df['bond'] == True, ['atom0', 'atom1', 'frame']]
    lbl0 = bonded['atom0'].map(lbls)
    lbl1 = bonded['atom1'].map(lbls)
    lbl = pd.concat((lbl0, lbl1), axis=1)
    lbl['frame'] = bonded['frame']
    bond_grps = lbl.groupby('frame')
    frames = df['frame'].unique().astype(np.int64)
    b0 = np.empty((len(frames), ), dtype='O')
    b1 = b0.copy()
    for i, frame in enumerate(frames):
        try: 
            b0[i] = bond_grps.get_group(frame)['atom0'].astype(np.int64).values
            b1[i] = bond_grps.get_group(frame)['atom1'].astype(np.int64).values
        except Exception:
            b0[i] = []
            b1[i] = []
    b0 = pd.Series(b0).to_json(orient='values')
    b1 = pd.Series(b1).to_json(orient='values')
    return {'two_b0': b0, 'two_b1': b1}

def frame_traits(df):
    """Get frame table traitlets."""
#    xi = Float().tag(sync=True)
#    xj = Float().tag(sync=True)
#    xk = Float().tag(sync=True)
#    yi = Float().tag(sync=True)
#    yj = Float().tag(sync=True)
#    yk = Float().tag(sync=True)
#    zi = Float().tag(sync=True)
#    zj = Float().tag(sync=True)
#    zk = Float().tag(sync=True)
#    ox = Float().tag(sync=True)
#    oy = Float().tag(sync=True)
#    oz = Float().tag(sync=True)
    return {}


@register
class UniverseScene(ExatomicScene):
    """A scene for viewing quantum systems."""
    _model_name = Unicode("UniverseSceneModel").tag(sync=True)
    _view_name = Unicode("UniverseSceneView").tag(sync=True)
    # Top level index
    frame_idx = Int(0).tag(sync=True)
    # Atom traits
    atom_x = Unicode().tag(sync=True)
    atom_y = Unicode().tag(sync=True)
    atom_z = Unicode().tag(sync=True)
    atom_s = Unicode().tag(sync=True)
    atom_r = Dict().tag(sync=True)
    atom_c = Dict().tag(sync=True)
    atom_3d = Bool(False).tag(sync=True)
    # Two traits
    two_b0 = Unicode().tag(sync=True)
    two_b1 = Unicode().tag(sync=True)
    # Field traits
    field_i = List().tag(sync=True)
    field_v = List().tag(sync=True)
    field_p = Dict().tag(sync=True)
    field_idx = Any().tag(sync=True)
    field_iso = Float(0.03).tag(sync=True)
    # Frame traits


@register
class UniverseWidget(ExatomicBox):
    """Test :class:`~exatomic.container.Universe` viewing widget."""
    _model_name = Unicode("UniverseWidgetModel").tag(sync=True)
    _view_name = Unicode("UniverseWidgetView").tag(sync=True)
    field_show = Bool(False).tag(sync=True)

    def init_gui(self, nframes=1, fields=None):
        super(UniverseWidget, self).init_gui(uni=True, test=False)
        playable = bool(nframes <= 1)
        frame_lims = {'min': 0, 'max': nframes-1, 'step': 1, 
                      'value': 0, 'layout': gui_lo}
        self.controls.update([('scn_frame', IntSlider(
                               description='Frame', **frame_lims)),
                              ('playing', Play(description="Press play", 
                               disabled=playable, **frame_lims)),
                              ('atom_3d', Button(description=" Atoms", 
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
                    def _field_options(c): self.scene.field_idx = c.new
                    self.controls['field_options'].observe(_field_options, names="value")
                else:
                    self.remove_field_gui()
                    self.controls.pop('field_options')

                self.gui = VBox([val for key, val in self.controls.items()],
                                 layout=Layout(width="200px"))
                self.set_gui()
            self.controls['field_show'].on_click(_field_show)

        def _scn_frame(c): self.scene.frame_idx = c.new
        def _atom_3d(b): self.scene.atom_3d = self.scene.atom_3d == False
        self.controls['scn_frame'].observe(_scn_frame, names='value')
        self.controls['atom_3d'].on_click(_atom_3d)
        jslink((self.controls['playing'], 'value'), (self.controls['scn_frame'], 'value'))
        self.gui = VBox([val for key, val in self.controls.items()],
                            layout=Layout(width="200px"))

        
    def __init__(self, uni, *args, **kwargs):
        #if not isinstance(uni, Universe):
        #    raise TypeError("Object passed to UniverseWidget must be a universe.")
        unargs = {}
        try: unargs.update(atom_traits(uni.atom))
        except AttributeError: pass
        try: unargs.update(two_traits(uni.atom_two, uni.atom.get_atom_labels()))
        except AttributeError: pass
        try: unargs.update(field_traits(uni.field))
        except AttributeError: pass
        try: unargs.update(frame_traits(uni.frame))
        except AttributeError: pass
        try: fields = ['null'] + unargs['field_i'][0]
        except KeyError: fields = None
        scene = UniverseScene(**unargs)
        self.init_gui(nframes=uni.atom.nframes, fields=fields)
        super(UniverseWidget, self).__init__(*args,
                                             scene=scene,
                                             **kwargs)

