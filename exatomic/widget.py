# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Universe Notebook Widget
#########################
"""
import os, re
import numpy as np
import pandas as pd
from glob import glob
from base64 import b64decode
from collections import OrderedDict
from traitlets import (Bool, Int, Float, Unicode,
                       List, Any, Dict, Instance)
from ipywidgets import (
    Box, VBox, HBox, FloatSlider, IntSlider, Play,
    IntRangeSlider,
    Widget, DOMWidget, Layout, Button, Dropdown,
    register, jslink, widget_serialization
)
from exa.relational.isotope import symbol_to_radius, symbol_to_color
from exatomic import __js_version__

###################
# Default layouts #
###################

gui_lo = Layout(width="195px")

#######################
# Common GUI patterns #
#######################


class Folder(VBox):
    """A VBox with a main widget that will show or hide widgets."""
    mlo = Layout(width="205px")
    lo = gui_lo


    def _close(self):
        """Close all widgets in the folder."""
        for widget in self.active_controls.values():
            widget.close()
        for widget in self.inactive_controls.values():
            widget.close()
        self.close()


    def _set_layout(self, widget=None):
        """Ensure all widgets are the same size."""
        if widget is not None:
            if isinstance(widget, Folder): return
            widget.layout = self.lo
        else:
            for widget in self.active_controls.values():
                if isinstance(widget, Folder): continue
                widget.layout = self.lo
            for widget in self.inactive_controls.values():
                if isinstance(widget, Folder): continue
                widget.layout = self.lo


    def _init_folder(self, control, content):
        """Set up controller for folder."""
        self.active_controls = OrderedDict([
            ('folder', control)])
        self.inactive_controls = content
        self._set_layout()
        def _controller(b):
            self.open_folder = self.open_folder == False
            self.set_gui()
        self.active_controls['folder'].on_click(_controller)


    def activate(self, *keys, update=False):
        """Activate a widget in the folder."""
        keys = list(self.inactive_controls.keys()) if not keys else keys
        for key in keys:
             self.active_controls[key] = self.inactive_controls.pop(key)
        if update:
            self.set_gui()


    def deactivate(self, *keys, update=False):
        """Deactivate a widget in the folder."""
        keys = list(self.active_controls.keys()) if not keys else keys
        for key in keys:
            if key == 'folder': continue
            self.inactive_controls[key] = self.active_controls.pop(key)
        if update:
            self.set_gui()


    def insert(self, idx, key, obj, active=True, update=False):
        """Insert a widget into the folder positionally by index."""
        self._set_layout(obj)
        nd = OrderedDict()
        od = self.active_controls if active else self.inactive_controls
        keys = list(od.keys())
        keys.insert(idx, key)
        for k in keys:
            if k == key: nd[k] = obj
            else: nd[k] = od[k]
        if active: self.active_controls = nd
        else: self.inactive_controls = nd
        if update:
            self.set_gui()


    def set_gui(self):
        """Open or close the folder."""
        self.children = list(self.active_controls.values())
        if not self.open_folder:
            self.children = self.children[:1]
        self.on_displayed(VBox._fire_children_displayed)


    def __init__(self, control, content, show=False, layout=None, **kwargs):
        self.open_folder = show
        self._init_folder(control, content)
        layout = self.mlo if layout is None else layout
        super(Folder, self).__init__(list(self.active_controls.values()),
                                     layout=layout, **kwargs)



def gui_field_widgets(uni=False, test=False):
    """New widgets for field GUI functionality."""
    flims = {"min": 30, "max": 60, "value": 30,
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
    return OrderedDict(iso=FloatSlider(**iso_lims),
                       nx=IntSlider(description="Nx", **flims),
                       ny=IntSlider(description="Ny", **flims),
                       nz=IntSlider(description="Nz", **flims))



################
# Base classes #
################
@register
class ExatomicScene(DOMWidget):
    """Resizable three.js scene."""
    _model_module_version = Unicode(__js_version__).tag(sync=True)
    _model_module_version = Unicode(__js_version__).tag(sync=True)
    _view_module = Unicode("jupyter-exatomic").tag(sync=True)
    _model_module = Unicode("jupyter-exatomic").tag(sync=True)
    _model_name = Unicode("ExatomicSceneModel").tag(sync=True)
    _view_name = Unicode("ExatomicSceneView").tag(sync=True)
    clear = Bool(False).tag(sync=True)
    save = Bool(False).tag(sync=True)
    field_pos = Unicode("003399").tag(sync=True)
    field_neg = Unicode("FF9900").tag(sync=True)
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
        typ = message['type']
        content = message['content']
        if typ == 'image': self._handle_image(content)

    def _handle_image(self, content):
        """Save a PNG of the scene."""
        savedir = self.savedir
        if not savedir: savedir = os.getcwd()
        nxt = 0
        fmt = '{:06d}.png'.format
        fname = self.imgname if self.imgname else fmt(nxt)
        while os.path.isfile(os.sep.join([savedir, fname])):
            nxt += 1
            fname = fmt(nxt)
        repl = 'data:image/png;base64,'
        with open(os.sep.join([savedir, fname]), 'wb') as f:
            f.write(b64decode(content.replace(repl, '')))

    def __init__(self, *args, **kwargs):
        lo = Layout(width="400", height="400")
        super(DOMWidget, self).__init__(*args, layout=lo, **kwargs)



@register
class PickerScene(ExatomicScene):
    _model_name = Unicode("PickerSceneModel").tag(sync=True)
    _view_name = Unicode("PickerSceneView").tag(sync=True)


@register
class HUDScene(ExatomicScene):
    _model_name = Unicode("HUDSceneModel").tag(sync=True)
    _view_name = Unicode("HUDSceneView").tag(sync=True)


@register
class ThreeAppScene(DOMWidget):
    """Resizable three.js scene."""
    _model_module_version = Unicode(__js_version__).tag(sync=True)
    _model_module_version = Unicode(__js_version__).tag(sync=True)
    _view_module = Unicode("jupyter-exatomic").tag(sync=True)
    _model_module = Unicode("jupyter-exatomic").tag(sync=True)
    _model_name = Unicode("ThreeAppSceneModel").tag(sync=True)
    _view_name = Unicode("ThreeAppSceneView").tag(sync=True)
    def __init__(self, *args, **kwargs):
        lo = Layout(width="400", height="400")
        super(DOMWidget, self).__init__(*args, layout=lo, **kwargs)

@register
class ExatomicBox(Box):
    """Base class for containers of a GUI and scene."""

    _model_module_version = Unicode(__js_version__).tag(sync=True)
    _model_module_version = Unicode(__js_version__).tag(sync=True)
    _model_module = Unicode("jupyter-exatomic").tag(sync=True)
    _view_module = Unicode("jupyter-exatomic").tag(sync=True)
    _model_name = Unicode("ExatomicBoxModel").tag(sync=True)
    _view_name = Unicode("ExatomicBoxView").tag(sync=True)
    scene = Instance(ExatomicScene, allow_none=True
                    ).tag(sync=True, **widget_serialization)


    def _close(self, b):
        """Shut down all active widgets within the container."""

        for widget in self.active_controls.values():
            try: widget._close()
            except: widget.close()

        for widget in self.inactive_controls.values():
            try: widget._close()
            except: widget.close()

        self.scene.close()
        self.close()


    def init_gui(self, uni=False, test=False):
        """Initialize generic GUI controls and register callbacks."""

        # Default GUI controls to control the scene
        self.inactive_controls = OrderedDict()
        self.active_controls = OrderedDict(
            close=Button(icon="trash", description=" Close", layout=gui_lo),
            clear=Button(icon="bomb", description=" Clear", layout=gui_lo),
            saves=Button(icon="camera", description=" Save", layout=gui_lo))

        def _clear(b):
            self.scene.clear = self.scene.clear == False
        def _saves(b):
            self.scene.save = self.scene.save == False

        self.active_controls['close'].on_click(self._close)
        self.active_controls['clear'].on_click(_clear)
        self.active_controls['saves'].on_click(_saves)

        # Inactive GUI controls common for subclasses
        fopts = gui_field_widgets(uni, test)

        def _iso(c): self.scene.field_iso = c.new
        def _nx(c): self.scene.field_nx = c.new
        def _ny(c): self.scene.field_ny = c.new
        def _nz(c): self.scene.field_nz = c.new

        fopts['iso'].observe(_iso, names='value')
        fopts['nx'].observe(_nx, names='value')
        fopts['ny'].observe(_ny, names='value')
        fopts['nz'].observe(_nz, names='value')

        self.inactive_controls['field'] = Folder(
                Button(description=' Fields', icon='cube'), fopts)


    def __init__(self, *args, scene=None, **kwargs):
        self.scene = ExatomicScene() if scene is None else scene
        if not hasattr(self, 'active_controls'): self.init_gui()
        self.gui = VBox(list(self.active_controls.values()))
        super(ExatomicBox, self).__init__(
                *args, children=[HBox([self.gui, self.scene])], **kwargs)



########################
# Basic example widget #
########################

@register
class TestScene(ExatomicScene):

    """A basic scene to test some javascript."""
    _model_name = Unicode("TestSceneModel").tag(sync=True)
    _view_name = Unicode("TestSceneView").tag(sync=True)
    field = Unicode("null").tag(sync=True)
    geom = Bool(True).tag(sync=True)



@register
class TestContainer(ExatomicBox):

    """A basic container to test some javascript."""
    _model_name = Unicode("TestContainerModel").tag(sync=True)
    _view_name = Unicode("TestContainerView").tag(sync=True)


    def init_gui(self):
        """Initialize specific GUI controls and register callbacks."""

        super(TestContainer, self).init_gui()

        geom = Button(icon="cubes", description=" Mesh", layout=gui_lo)
        def _geom(b): self.scene.geom = self.scene.geom == False
        geom.on_click(_geom)
        self.active_controls['geom'] = geom

        fopts = ['null', 'sphere', 'torus', 'ellipsoid']
        fopts = Dropdown(options=fopts, layout=gui_lo)
        def _field(c): self.scene.field = c.new
        fopts.observe(_field, names='value')

        folder = self.inactive_controls.pop('field')
        folder.insert(1, 'options', fopts)
        folder.activate('iso', 'nx', 'ny', 'nz')
        self.active_controls['field'] = folder


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

        uni_field_lists = OrderedDict([
            ('Hydrogenic', ['1s',   '2s',   '2px', '2py', '2pz',
                            '3s',   '3px',  '3py', '3pz',
                            '3d-2', '3d-1', '3d0', '3d+1', '3d+2']),
            ('Gaussian', ['s', 'px', 'py', 'pz', 'd200', 'd110',
                          'd101', 'd020', 'd011', 'd002', 'f300',
                          'f210', 'f201', 'f120', 'f111', 'f102',
                          'f030', 'f021', 'f012', 'f003']),
            ('SolidHarmonic', [str(i) for i in range(8)])
        ])

        folder = self.inactive_controls.pop('field')
        fopts = list(uni_field_lists.keys())
        fopts = Dropdown(options=fopts, layout=gui_lo)
        field_widgets = [(key, Dropdown(options=val, layout=gui_lo))
                         for key, val in uni_field_lists.items()]
        ml_widgets = [(str(l), Dropdown(options=range(-l, l+1),
                      layout=gui_lo)) for l in range(8)]
        self.inactive_controls.update(field_widgets)
        self.inactive_controls.update(ml_widgets)
        fkind = self.inactive_controls[self.scene.field]

        def _field(c):
            self.scene.field = c.new
            fk = uni_field_lists[c.new][0]
            self.scene.field_kind = fk

            if self.scene.field == 'SolidHarmonic':
                folder.insert(3, 'fml', self.inactive_controls[fk])
                folder.active_controls.pop('fkind')
            elif 'fml' in folder.active_controls:
                folder.active_controls.pop('fml')

            folder.insert(2, 'fkind', self.inactive_controls[c.new])
            folder.set_gui()
        fopts.observe(_field, names="value")

        def _field_kind(c):
            self.scene.field_kind = c.new
            if self.scene.field == 'SolidHarmonic':
                self.scene.field_ml = self.inactive_controls[c.new].options[0]
                folder.insert(3, 'fml', self.inactive_controls[c.new],
                              update=True)
            elif 'fml' in folder.active_controls:
                folder.deactivate('fml', update=True)
        for key, widget in field_widgets:
            widget.observe(_field_kind, names='value')

        def _field_ml(c):
            self.scene.field_ml = c.new
        for key, widget in ml_widgets:
            widget.observe(_field_ml, names='value')

        folder.insert(1, 'fopts', fopts)
        folder.insert(2, 'fkind', fkind)
        folder.activate('iso', 'nx', 'ny', 'nz')
        self.active_controls['field'] = folder


    def __init__(self, *args, **kwargs):
        super(TestUniverse, self).__init__(
                *args, scene=TestUniverseScene(), **kwargs)


################################
# Universe and related widgets #
################################

def atom_traits(df):
    """
    Get atom table traitlets. Atomic size (using the covalent radius) and atom
    colors (using the common `Jmol`_ color scheme) are packed as dicts and
    obtained from the static data in exa.

    .. _Jmol: http://jmol.sourceforge.net/jscolors/
    """
    traits = {}
    if 'label' in df.columns: df['l'] = df['label']
    elif 'tag' in df.columns: df['l'] = df['tag']
    else: df['l'] = df['symbol'] + df.index.astype(str)
    grps = df.groupby('frame')
    for col in ['x', 'y', 'z', 'l']:
        traits['atom_'+col] = grps.apply(
            lambda y: y[col].to_json(
            orient='values', double_precision=3)
            ).to_json(orient="values").replace('"', '')
    del df['l']
    repl = {r'\\': '', '"\[': '[', '\]"': ']'}
    replpat = re.compile('|'.join(repl.keys()))
    repl = {'\\': '', '"[': '[', ']"': ']'}
    traits['atom_l'] = replpat.sub(lambda m: repl[m.group(0)], traits['atom_l'])
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
    #vals = [f.tolist() for f in df.field_values]
    vals = '[' + ','.join([f.to_json(orient='values',
                           double_precision=5) for f in df.field_values]) + ']'
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
    field_v = Unicode().tag(sync=True)
    field_p = Dict().tag(sync=True)
    field_idx = Any().tag(sync=True)
    field_iso = Float(0.03).tag(sync=True)
    field_show = Bool(False).tag(sync=True)
    cont_show = Bool(False).tag(sync=True)
    cont_axis = Unicode("z").tag(sync=True)
    cont_num = Int(10).tag(sync=True)
    cont_lim = List([-8, -1]).tag(sync=True)
    cont_val = Float(0.0).tag(sync=True)
    # Frame traits


@register
class UniverseWidget(ExatomicBox):
    """Test :class:`~exatomic.container.Universe` viewing widget."""
    _model_name = Unicode("UniverseWidgetModel").tag(sync=True)
    _view_name = Unicode("UniverseWidgetView").tag(sync=True)

    def init_gui(self, nframes=1, fields=None):

        super(UniverseWidget, self).init_gui(uni=True, test=False)

        atoms = Button(description=' Atoms', icon='adjust', layout=gui_lo)
        def _atom_3d(b): self.scene.atom_3d = self.scene.atom_3d == False
        atoms.on_click(_atom_3d)
        self.active_controls['atom_3d'] = atoms

        playable = bool(nframes <= 1)
        flims = dict(min=0, max=nframes-1, step=1, value=0, layout=gui_lo)
        control = Button(description=' Animate', icon='play')
        content = OrderedDict([
            ('playing', Play(disabled=playable, **flims)),
            ('scn_frame', IntSlider(description='Frame', **flims))
        ])
        def _scn_frame(c): self.scene.frame_idx = c.new
        content['scn_frame'].observe(_scn_frame, names='value')
        jslink((content['playing'], 'value'),
               (content['scn_frame'], 'value'))
        self.active_controls['frame'] = Folder(control, content)
        self.active_controls['frame'].activate()


        if fields is not None:
            # Main field folder
            folder = self.inactive_controls.pop('field')
            fopts = Dropdown(options=fields, layout=gui_lo)
            def _fopts(c): self.scene.field_idx = c.new
            fopts.observe(_fopts, names='value')
            # Make an isosurface folder
            isos = Button(description=' Isosurfaces', icon='cube')
            def _fshow(b):
                self.scene.field_show = self.scene.field_show == False
            isos.on_click(_fshow)
            # Move the isosurface button to the subfolder
            iso = folder.inactive_controls.pop('iso')
            isofolder = Folder(isos, OrderedDict([
                ('fopts', fopts),
                ('iso', iso)
            ]), layout=Layout(width="200px"))
            isofolder.activate()
            folder.insert(1, 'iso', isofolder, update=True)
            # Make a contour folder
            control = Button(description=' Contours', icon='dot-circle-o')
            def _cshow(b):
                self.scene.cont_show = self.scene.cont_show == False
            control.on_click(_cshow)
            content = OrderedDict([
                ('fopts', fopts),
                ('axis', Dropdown(options=['x', 'y', 'z'], value='z')),
                ('num', IntSlider(description='N', min=5, max=20,
                                  value=10, step=1, layout=gui_lo,)),
                ('lim', IntRangeSlider(description="10**Limits", min=-10,
                                       max=0, step=1, value=[-8, -1],)),
                ('val', FloatSlider(description="Value",
                                    min=-5, max=5, value=0,)),
            ])
            def _cont_axis(c): self.scene.cont_axis = c.new
            def _cont_num(c): self.scene.cont_num = c.new
            def _cont_lim(c): self.scene.cont_lim = c.new
            def _cont_val(c): self.scene.cont_val = c.new
            content['axis'].observe(_cont_axis, names='value')
            content['num'].observe(_cont_num, names='value')
            content['lim'].observe(_cont_lim, names='value')
            content['val'].observe(_cont_val, names='value')
            contour = Folder(control, content, layout=Layout(width="200px"))
            contour.activate()
            folder.insert(2, 'contour', contour, update=True)
            self.active_controls['field'] = folder


    def __init__(self, uni, *args, **kwargs):
        unargs = {}
        try: unargs.update(atom_traits(uni.atom))
        except AttributeError: pass
        try: unargs.update(two_traits(uni.atom_two,
                           uni.atom.get_atom_labels()))
        except AttributeError: pass
        try: unargs.update(frame_traits(uni.frame))
        except AttributeError: pass
        try:
            unargs.update(field_traits(uni.field))
            fields = ['null'] + unargs['field_i'][0]
        except AttributeError:
            fields = None
        scene = UniverseScene(**unargs)
        self.init_gui(nframes=uni.atom.nframes, fields=fields)
        super(UniverseWidget, self).__init__(*args,
                                             scene=scene,
                                             **kwargs)
