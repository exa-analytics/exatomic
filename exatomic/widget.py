# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Universe Notebook Widget
#########################
"""

from ipywidgets import Button, Dropdown

from .widget_base import ExatomicScene, ExatomicBox
from .widget_utils import _wlo, _ListDict, _scene_grid
from .traits import atom_traits, field_traits, two_traits, frame_traits

class UniverseWidget:
    pass


class TestContainer(ExatomicBox):


    def _field_folder(self):
        folder = super(TestContainer, self)._field_folder()
        fopts = ['null', 'Sphere', 'Torus', 'Ellipsoid']
        fopts = Dropdown(options=fopts)
        fopts.active = True
        fopts.disabled = False
        def _field(c):
            for idx in self.active_scene_indices:
                self.scenes[idx].field = c.new
        fopts.observe(_field, names='value')
        folder.insert(1, 'options', fopts)
        return folder


    def _init_gui(self):
        mainopts = super(TestContainer, self)._init_gui()
        geom = Button(icon="gear", description=" Mesh", layout=_wlo)
        def _geom(b):
            for idx in self.active_scene_indices:
                self.scenes[idx].geom = self.scenes[idx].geom == False
        geom.on_click(_geom)
        mainopts.update([('geom', geom),
                         ('field', self._field_folder())])
        return mainopts


    def __init__(self, scenes, **kwargs):
        self.uni = False
        self.test = True
        super(TestContainer, self).__init__(scenes, **kwargs)


class TestUniverse(ExatomicBox):


    def _field_folder(self):
        for scn in self.scenes:
            for attr in ['field_ox', 'field_oy', 'field_oz']:
                setattr(scn, attr, -30.0)
            for attr in ['field_fx', 'field_fy', 'field_fz']:
                setattr(scn, attr, 30.0)
            scn.field = 'Hydrogenic'
            scn.field_iso = 0.0005
            scn.field_kind = '1s'
        folder = super(TestUniverse, self)._field_folder()
        uni_field_lists = _ListDict([
            ('Hydrogenic', ['1s',   '2s',   '2px', '2py', '2pz',
                            '3s',   '3px',  '3py', '3pz',
                            '3d-2', '3d-1', '3d0', '3d+1', '3d+2']),
            ('Gaussian', ['s', 'px', 'py', 'pz', 'd200', 'd110',
                          'd101', 'd020', 'd011', 'd002', 'f300',
                          'f210', 'f201', 'f120', 'f111', 'f102',
                          'f030', 'f021', 'f012', 'f003']),
            ('SolidHarmonic', [str(i) for i in range(8)])])
        field_widgets = _ListDict([
            (key, Dropdown(options=vals))
            for key, vals in uni_field_lists.items()])
        ml_widgets = _ListDict([
            (str(l), Dropdown(options=[str(i) for i in range(-l, l+1)]))
            for l in range(8)])
        fopts = list(uni_field_lists.keys())
        folder.update(field_widgets, relayout=True)
        folder.update(ml_widgets, relayout=True)

        def _field(c):
            folder.deactivate(c.old)
            folder.activate(c.new, enable=True, update=True)
            fk = uni_field_lists[c.new][0]
            if c.new == 'SolidHarmonic':
                folder.activate(fk, enable=True, update=True)
            elif c.old == 'SolidHarmonic':
                folder.deactivate(self.scenes[0].field_kind, update=True)
            for idx in self.active_scene_indices:
                self.scenes[idx].field = c.new
            for idx in self.active_scene_indices:
                self.scenes[idx].field_kind = fk

        def _field_kind(c):
            for idx in self.active_scene_indices:
                self.scenes[idx].field_kind = c.new
            print(c, self.scenes[0].field_ml)
            if self.scenes[0].field == 'SolidHarmonic':
                for idx in self.active_scene_indices:
                    self.scenes[idx].field_ml = folder[c.new].options[0]
                folder.deactivate(c.old)
                folder.activate(c.new, enable=True, update=True)

        def _field_ml(c):
            for idx in self.active_scene_indices:
                self.scenes[idx].field_ml = c.new

        for key, obj in field_widgets.items():
            folder.deactivate(key)
            obj.observe(_field_kind, names='value')

        for key, obj in ml_widgets.items():
            folder.deactivate(key)
            obj.observe(_field_ml, names='value')

        fopts = Dropdown(options=fopts)
        fopts.observe(_field, names="value")
        folder.insert(1, 'fopts', fopts)
        folder.activate('Hydrogenic', enable=True, update=True)
        folder.move_to_end('alpha', 'iso', 'nx', 'ny', 'nz')
        return folder


    def _init_gui(self):
        mainopts = super(TestUniverse, self)._init_gui()
        mainopts.update([('field', self._field_folder())])
        return mainopts


    def __init__(self, scenes, **kwargs):
        self.uni = True
        self.test = True
        super(TestUniverse, self).__init__(scenes, **kwargs)
        # self.field = Unicode("null").tag(sync=True)
        # field_kind = Unicode("").tag(sync=True)

# TestUniverse
#     def _init_gui(self):
#         super(TestUniverse, self)._init_gui(uni=True, test=True)
#

#
# @register
#
# class TestScene(SandboxScene):
#     """A basic scene to test some javascript."""
#
#     _model_name = Unicode("TestSceneModel").tag(sync=True)
#     _view_name = Unicode("TestSceneView").tag(sync=True)
#     field = Unicode("null").tag(sync=True)
#     geom = Bool(True).tag(sync=True)
#


# class TestContainer(ExatomicBox):
#     """A basic container to test some javascript and GUI."""
#
#     def _init_gui(self):
#         """Initialize specific GUI controls and register callbacks."""
#
#         super(TestContainer, self)._init_gui()
#
#         geom = Button(icon="cubes", description=" Mesh", layout=gui_lo)
#         def _geom(b): self.scene.geom = self.scene.geom == False
#         geom.on_click(_geom)
#         self.active_controls['geom'] = geom
#
#         fopts = ['null', 'Sphere', 'Torus', 'Ellipsoid']
#         fopts = Dropdown(options=fopts, layout=gui_lo)
#         def _field(c): self.scene.field = c.new
#         fopts.observe(_field, names='value')
#
#         folder = self.inactive_controls.pop('field')
#         folder.insert(1, 'options', fopts)
#         folder.activate('iso', 'nx', 'ny', 'nz')
#         self.active_controls['field'] = folder
#
#     def __init__(self, *args, **kwargs):
#         super(TestContainer, self).__init__(*args,
#                                             scene=TestScene(),
#                                             **kwargs)


## OLD ###########################################################3
## OLD ###########################################################3
## OLD ###########################################################3
## OLD ###########################################################3
## OLD ###########################################################3
## OLD ###########################################################3

# gui_lo = Layout(width="195px")
#
# #######################
# # Common GUI patterns #
# #######################
#
#
# class Folder(VBox):
#     """A VBox with a main widget that will show or hide widgets."""
#     mlo = Layout(width="205px")
#     lo = gui_lo
#     lo.width = "175px"
#     lo.left = "20px"
#
#
#     def _close(self):
#         """Close all widgets in the folder."""
#         for widget in self.active_controls.values():
#             widget.close()
#         for widget in self.inactive_controls.values():
#             widget.close()
#         self.close()
#
#
#     def _set_layout(self, widget=None):
#         """Ensure all widgets are the same size."""
#         if widget is not None:
#             if isinstance(widget, Folder): return
#             widget.layout = self.lo
#         else:
#             for widget in self.active_controls.values():
#                 if isinstance(widget, Folder): continue
#                 widget.layout = self.lo
#             for widget in self.inactive_controls.values():
#                 if isinstance(widget, Folder): continue
#                 widget.layout = self.lo
#
#
#     def _init_folder(self, control, content):
#         """Set up controller for folder."""
#         self.active_controls = _ListDict([
#             ('folder', control)])
#         self.inactive_controls = content
#         self._set_layout()
#         def _controller(b):
#             self.open_folder = self.open_folder == False
#             self.set_gui()
#         self.active_controls['folder'].tooltip = 'Expand'
#         self.active_controls['folder'].on_click(_controller)
#
#
#     def activate(self, *keys, **kwargs):
#         """Activate a widget in the folder."""
#         update = kwargs.pop("update", False)
#         keys = list(self.inactive_controls.keys()) if not keys else keys
#         for key in keys:
#              self.active_controls[key] = self.inactive_controls.pop(key)
#         if update:
#             self.set_gui()
#
#
#     def deactivate(self, *keys, **kwargs):
#         """Deactivate a widget in the folder."""
#         update = kwargs.pop("update", False)
#         keys = list(self.active_controls.keys()) if not keys else keys
#         for key in keys:
#             if key == 'folder': continue
#             self.inactive_controls[key] = self.active_controls.pop(key)
#         if update:
#             self.set_gui()
#
#
#     def insert(self, idx, key, obj, active=True, update=False):
#         """Insert a widget into the folder positionally by index."""
#         self._set_layout(obj)
#         nd = ODict()
#         od = self.active_controls if active else self.inactive_controls
#         keys = list(od.keys())
#         keys.insert(idx, key)
#         for k in keys:
#             if k == key: nd[k] = obj
#             else: nd[k] = od[k]
#         if active: self.active_controls = nd
#         else: self.inactive_controls = nd
#         if update:
#             self.set_gui()
#
#
#     def set_gui(self):
#         """Open or close the folder."""
#         self.children = list(self.active_controls.values())
#         if not self.open_folder:
#             self.children[0].tooltip = 'Expand'
#             self.children = self.children[:1]
#         else:
#             self.children[0].tooltip = 'Collapse'
#         self.on_displayed(VBox._fire_children_displayed)
#
#
#     def __init__(self, control, content, **kwargs):
#         show = kwargs.pop("show", False)
#         layout = kwargs.pop("layout", None)
#         self.open_folder = show
#         self._init_folder(control, content)
#         layout = self.mlo if layout is None else layout
#         self._set_layout()
#         super(Folder, self).__init__(list(self.active_controls.values()),
#                                      layout=layout, **kwargs)


#
#
#
# ################
# # Base classes #
# ################
# @register
# class ExatomicScene(DOMWidget):
#     """Resizable three.js scene."""
#     _model_module_version = Unicode(__js_version__).tag(sync=True)
#     _model_module_version = Unicode(__js_version__).tag(sync=True)
#     _view_module = Unicode("jupyter-exatomic").tag(sync=True)
#     _model_module = Unicode("jupyter-exatomic").tag(sync=True)
#     _model_name = Unicode("ExatomicSceneModel").tag(sync=True)
#     _view_name = Unicode("ExatomicSceneView").tag(sync=True)
#     clear = Bool(False).tag(sync=True)
#     save = Bool(False).tag(sync=True)
#     save_cam = Bool(False).tag(sync=True)
#     cameras = List(trait=Dict()).tag(sync=True)
#     field_pos = Unicode("#003399").tag(sync=True)
#     field_neg = Unicode("#FF9900").tag(sync=True)
#     field_iso = Float(2.0).tag(sync=True)
#     field_ox = Float(-3.0).tag(sync=True)
#     field_oy = Float(-3.0).tag(sync=True)
#     field_oz = Float(-3.0).tag(sync=True)
#     field_fx = Float(3.0).tag(sync=True)
#     field_fy = Float(3.0).tag(sync=True)
#     field_fz = Float(3.0).tag(sync=True)
#     field_nx = Int(31).tag(sync=True)
#     field_ny = Int(31).tag(sync=True)
#     field_nz = Int(31).tag(sync=True)
#     savedir = Unicode().tag(sync=True)
#     imgname = Unicode().tag(sync=True)
#
#     def _handle_custom_msg(self, msg, callback):
#         """Custom message handler."""
#         if msg['type'] == 'image':
#             self._handle_image(msg['content'])
#         elif msg['type'] == 'camera':
#             self._handle_camera(msg['content'])
#         else: print("Custom msg not handled.\n"
#                     "type of msg : {}\n"
#                     "msg         : {}".format(msg['type'], msg['content']))
#
#     def _handle_camera(self, content):
#         self.cameras.append(content)
#
#     def _handle_image(self, content):
#         """Save a PNG of the scene."""
#         savedir = self.savedir
#         if not savedir: savedir = os.getcwd()
#         nxt = 0
#         fmt = '{:06d}.png'.format
#         fname = self.imgname if self.imgname else fmt(nxt)
#         while os.path.isfile(os.sep.join([savedir, fname])):
#             nxt += 1
#             fname = fmt(nxt)
#         repl = 'data:image/png;base64,'
#         with open(os.sep.join([savedir, fname]), 'wb') as f:
#             f.write(b64decode(content.replace(repl, '')))
#
#     def _set_camera(self, c):
#         if c.new == -1: return
#         self.send({'type': 'camera',
#                    'content': self.cameras[c.new]})
#
#     def _close(self):
#         self.send({'type': 'close'})
#         self.close()
#
#     def __init__(self, *args, **kwargs):
#         self.cameras = []
#         # lo = Layout(width="400", height="400", flex="1 1 auto")
#         lo = kwargs.pop('layout', None)
#         if lo is None:
#             lo = Layout(display='flex', width='100%',
#                         align_items='stretch') #width="100%", height="100%", flex="1 1 auto")
#         super(DOMWidget, self).__init__(*args, layout=lo, **kwargs)
#
#
# @register
# class ExatomicBox(Box):
#     """Base class for containers of a GUI and scene."""
#
#     _model_module_version = Unicode(__js_version__).tag(sync=True)
#     _model_module_version = Unicode(__js_version__).tag(sync=True)
#     _model_module = Unicode("jupyter-exatomic").tag(sync=True)
#     _view_module = Unicode("jupyter-exatomic").tag(sync=True)
#     _model_name = Unicode("ExatomicBoxModel").tag(sync=True)
#     _view_name = Unicode("ExatomicBoxView").tag(sync=True)
#     #scene = Instance(ExatomicScene, allow_none=True
#     #                ).tag(sync=True, **widget_serialization)
#
#
#     def _close(self, b):
#         """Shut down all active widgets within the container."""
#         for widget in self.active_controls.values():
#             try: widget._close()
#             except: widget.close()
#
#         for widget in self.inactive_controls.values():
#             try: widget._close()
#             except: widget.close()
#
#         self.scene._close()
#         self.close()
#
#
#     def _init_gui(self, uni=False, test=False):
#         """Initialize generic GUI controls and register callbacks."""
#
#         # Default GUI controls to control the scene
#         self.inactive_controls = OrderedDict()
#         self.active_controls = OrderedDict(
#             close=Button(icon='trash', description=' Close', layout=gui_lo),
#             clear=Button(icon='bomb', description=' Clear', layout=gui_lo))
#
#         copts = OrderedDict([
#             ('get', Button(icon='arrow-circle-down', description=' Save')),
#             ('set', IntSlider(description='Load', min=-1,
#                               max=len(self.scene.cameras) - 1,
#                               value=-1, step=1))])
#         def _save_camera(b):
#             self.scene.save_cam = self.scene.save_cam == False
#             (self.active_controls['camera']
#                  .active_controls['set'].max) = len(self.scene.cameras)
#         copts['get'].on_click(_save_camera)
#         copts['set'].observe(self.scene._set_camera, names='value')
#         cfolder = Folder(Button(icon='camera', description=' Camera'), copts)
#         cfolder.activate()
#
#         self.active_controls['camera'] = cfolder
#         self.active_controls['saves'] = Button(
#             icon='save', description=' Save', layout=gui_lo)
#
#         def _clear(b):
#             self.scene.clear = self.scene.clear == False
#         def _saves(b):
#             self.scene.save = self.scene.save == False
#
#         self.active_controls['close'].on_click(self._close)
#         self.active_controls['clear'].on_click(_clear)
#         self.active_controls['saves'].on_click(_saves)
#
#         # Inactive GUI controls common for subclasses
#         fopts = gui_field_widgets(uni, test)
#
#         def _iso(c): self.scene.field_iso = c.new
#         def _nx(c): self.scene.field_nx = c.new
#         def _ny(c): self.scene.field_ny = c.new
#         def _nz(c): self.scene.field_nz = c.new
#
#         fopts['iso'].observe(_iso, names='value')
#         fopts['nx'].observe(_nx, names='value')
#         fopts['ny'].observe(_ny, names='value')
#         fopts['nz'].observe(_nz, names='value')
#
#         self.inactive_controls['field'] = Folder(
#                 Button(description=' Fields', icon='cube'), fopts)
#
#
#     def __init__(self, scenekwargs=None, *args, **kwargs):
#         scenekwargs = {} if scenekwargs is None else scenekwargs
#
#         if not hasattr(self, 'scene'):
#             self.scene = ExatomicScene(**scenekwargs)
#         elif 'scene' in kwargs:
#             self.scene = kwargs.pop('scene')
#
#         if not hasattr(self, 'active_controls'): self._init_gui()
#         self.gui = VBox(list(self.active_controls.values()))
#
#         children = kwargs.pop('children', None)
#         if children is None: children = [self.gui, self.scene]
#         lo = Layout(width="100%", height="100%")
#         super(ExatomicBox, self).__init__(
#             *args, children=children, layout=lo, **kwargs)
#
#
#
# ########################
# # Basic example widget #
# ########################
#
# @register
# class TestScene(ExatomicScene):
#     """A basic scene to test some javascript."""
#
#     _model_name = Unicode("TestSceneModel").tag(sync=True)
#     _view_name = Unicode("TestSceneView").tag(sync=True)
#     field = Unicode("null").tag(sync=True)
#     geom = Bool(True).tag(sync=True)
#
#
#
# class TestContainer(ExatomicBox):
#     """A basic container to test some javascript and GUI."""
#
#     def _init_gui(self):
#         """Initialize specific GUI controls and register callbacks."""
#
#         super(TestContainer, self)._init_gui()
#
#         geom = Button(icon="cubes", description=" Mesh", layout=gui_lo)
#         def _geom(b): self.scene.geom = self.scene.geom == False
#         geom.on_click(_geom)
#         self.active_controls['geom'] = geom
#
#         fopts = ['null', 'Sphere', 'Torus', 'Ellipsoid']
#         fopts = Dropdown(options=fopts, layout=gui_lo)
#         def _field(c): self.scene.field = c.new
#         fopts.observe(_field, names='value')
#
#         folder = self.inactive_controls.pop('field')
#         folder.insert(1, 'options', fopts)
#         folder.activate('iso', 'nx', 'ny', 'nz')
#         self.active_controls['field'] = folder
#
#     def __init__(self, *args, **kwargs):
#         super(TestContainer, self).__init__(*args,
#                                             scene=TestScene(),
#                                             **kwargs)
#
#
# ###########################
# # Universe example widget #
# ###########################
#
# @register
# class TestUniverseScene(ExatomicScene):
#     """Test :class:`~exatomic.container.Universe` scene."""
#
#     _model_name = Unicode("TestUniverseSceneModel").tag(sync=True)
#     _view_name = Unicode("TestUniverseSceneView").tag(sync=True)
#     field_iso = float(0.0005).tag(sync=true)
#     field = Unicode('Hydrogenic').tag(sync=True)
#     field_kind = Unicode('1s').tag(sync=True)
#     field_ox = Float(-30.0).tag(sync=True)
#     field_oy = Float(-30.0).tag(sync=True)
#     field_oz = Float(-30.0).tag(sync=True)
#     field_fx = Float(30.0).tag(sync=True)
#     field_fy = Float(30.0).tag(sync=True)
#     field_fz = Float(30.0).tag(sync=True)
#     field_ml = Int(0).tag(sync=True)
#
#
# class TestUniverse(ExatomicBox):
#     """Test :class:`~exatomic.container.Universe` test widget."""
#
#     def _init_gui(self):
#         super(TestUniverse, self)._init_gui(uni=True, test=True)
#
#         uni_field_lists = OrderedDict([
#             ('Hydrogenic', ['1s',   '2s',   '2px', '2py', '2pz',
#                             '3s',   '3px',  '3py', '3pz',
#                             '3d-2', '3d-1', '3d0', '3d+1', '3d+2']),
#             ('Gaussian', ['s', 'px', 'py', 'pz', 'd200', 'd110',
#                           'd101', 'd020', 'd011', 'd002', 'f300',
#                           'f210', 'f201', 'f120', 'f111', 'f102',
#                           'f030', 'f021', 'f012', 'f003']),
#             ('SolidHarmonic', [str(i) for i in range(8)])
#         ])
#
#         folder = self.inactive_controls.pop('field')
#         fopts = list(uni_field_lists.keys())
#         fopts = Dropdown(options=fopts, layout=gui_lo)
#         field_widgets = [(key, Dropdown(options=val, layout=gui_lo))
#                          for key, val in uni_field_lists.items()]
#         ml_widgets = [(str(l), Dropdown(options=range(-l, l+1),
#                       layout=gui_lo)) for l in range(8)]
#         self.inactive_controls.update(field_widgets)
#         self.inactive_controls.update(ml_widgets)
#         fkind = self.inactive_controls[self.scene.field]
#
#         def _field(c):
#             self.scene.field = c.new
#             fk = uni_field_lists[c.new][0]
#             self.scene.field_kind = fk
#             if self.scene.field == 'SolidHarmonic':
#                 folder.insert(3, 'fml', self.inactive_controls[fk])
#                 folder.active_controls.pop('fkind')
#             elif 'fml' in folder.active_controls:
#                 folder.active_controls.pop('fml')
#
#             folder.insert(2, 'fkind', self.inactive_controls[c.new])
#             folder.set_gui()
#
#         fopts.observe(_field, names="value")
#
#         def _field_kind(c):
#             self.scene.field_kind = c.new
#             if self.scene.field == 'SolidHarmonic':
#                 self.scene.field_ml = self.inactive_controls[c.new].options[0]
#                 folder.insert(3, 'fml', self.inactive_controls[c.new],
#                               update=True)
#             elif 'fml' in folder.active_controls:
#                 folder.deactivate('fml', update=True)
#
#         for key, widget in field_widgets:
#             widget.observe(_field_kind, names='value')
#
#         def _field_ml(c):
#             self.scene.field_ml = c.new
#         for key, widget in ml_widgets:
#             widget.observe(_field_ml, names='value')
#
#         folder.insert(1, 'fopts', fopts)
#         folder.insert(2, 'fkind', fkind)
#         folder.activate('iso', 'nx', 'ny', 'nz')
#         self.active_controls['field'] = folder
#
#
#     def __init__(self, scenekwargs=None, *args, **kwargs):
#         scenekwargs = {} if scenekwargs is None else scenekwargs
#         super(TestUniverse, self).__init__(
#             *args, scene=TestUniverseScene(**scenekwargs), **kwargs)
#
#
# ################################
# # Universe and related widgets #
# ################################
#
# def atom_traits(df, atomcolors=None, atomradii=None):
#     """
#     Get atom table traits. Atomic size (using the covalent radius) and atom
#     colors (using the common `Jmol`_ color scheme) are packed as dicts and
#     obtained from the static data in exa.
#
#     .. _Jmol: http://jmol.sourceforge.net/jscolors/
#     """
#     # Implement logic to automatically choose
#     # whether or not to create labels
#     labels = True
#     atomcolors = pd.Series() if atomcolors is None else pd.Series(atomcolors)
#     atomradii = pd.Series() if atomradii is None else pd.Series(atomradii)
#     traits = {}
#     cols = ['x', 'y', 'z']
#     if labels:
#         cols.append('l')
#         if 'tag' in df.columns: df['l'] = df['tag']
#         else: df['l'] = df['symbol'] + df.index.astype(str)
#     grps = df.groupby('frame')
#     for col in cols:
#         ncol = 'atom_' + col
#         if col == 'l':
#             labels = grps.apply(lambda y: y[col].to_json(orient='values')
#                 ).to_json(orient="values")
#             repl = {r'\\': '', '"\[': '[', '\]"': ']'}
#             replpat = re.compile('|'.join(repl.keys()))
#             repl = {'\\': '', '"[': '[', ']"': ']'}
#             traits['atom_l'] = replpat.sub(lambda m: repl[m.group(0)],
#                                            labels)
#             del df['l']
#         else:
#             traits[ncol] = grps.apply(
#                 lambda y: y[col].to_json(
#                 orient='values', double_precision=3)
#                 ).to_json(orient="values").replace('"', '')
#     syms = grps.apply(lambda g: g['symbol'].cat.codes.values)
#     symmap = {i: v for i, v in enumerate(df['symbol'].cat.categories)
#               if v in df.unique_atoms}
#     unq = df['symbol'].astype(str).unique()
#     radii = {k: sym2radius[k] for k in unq}
#     colors = {k: sym2color[k] for k in unq}
#     colors.update(atomcolors)
#     radii.update(atomradii)
#     traits['atom_s'] = syms.to_json(orient='values')
#     traits['atom_r'] = {i: 0.5 * radii[v] for i, v in symmap.items()}
#     traits['atom_c'] = {i: colors[v] for i, v in symmap.items()}
#     return traits
#
# def field_traits(df):
#     """Get field table traits."""
#     df['frame'] = df['frame'].astype(int)
#     df['nx'] = df['nx'].astype(int)
#     df['ny'] = df['ny'].astype(int)
#     df['nz'] = df['nz'].astype(int)
#     if not all((col in df.columns for col in ['fx', 'fy', 'fz'])):
#         for d, l in [('x', 'i'), ('y', 'j'), ('z', 'k')]:
#             df['f'+d] = df['o'+d] + (df['n'+d] - 1) * df['d'+d+l]
#     grps = df.groupby('frame')
#     fps = grps.apply(lambda x: x[['ox', 'oy', 'oz',
#                                   'nx', 'ny', 'nz',
#                                   'fx', 'fy', 'fz']].T.to_dict()).to_dict()
#     try: idxs = list(map(list, grps.groups.values()))
#     except: idxs = [list(grp.index) for i, grp in grps]
#     #vals = [f.tolist() for f in df.field_values]
#     # shape0 = len(df.field_values)
#     # shape1 = len(df.field_values[0])
#     # vals = np.empty((shape0, shape1), dtype=np.float32)
#     # for i in range(shape0):
#     #     vals[i] = df.field_values[i].values
#     vals = '[' + ','.join([f.to_json(orient='values',
#                            double_precision=5) for f in df.field_values]) + ']'
#     return {'field_v': vals, 'field_i': idxs, 'field_p': fps}
#
# #def two_traits(df, lbls):
# def two_traits(uni):
#     """Get two table traitlets."""
#     if not hasattr(uni, "atom_two"):
#         raise AttributeError("for the catcher")
#     if "frame" not in uni.atom_two.columns:
#         uni.atom_two['frame'] = uni.atom_two['atom0'].map(uni.atom['frame'])
#     lbls = uni.atom.get_atom_labels()
#     df = uni.atom_two
#     bonded = df.loc[df['bond'] == True, ['atom0', 'atom1', 'frame']]
#     lbl0 = bonded['atom0'].map(lbls)
#     lbl1 = bonded['atom1'].map(lbls)
#     lbl = pd.concat((lbl0, lbl1), axis=1)
#     lbl['frame'] = bonded['frame']
#     bond_grps = lbl.groupby('frame')
#     frames = df['frame'].unique().astype(np.int64)
#     b0 = np.empty((len(frames), ), dtype='O')
#     b1 = b0.copy()
#     for i, frame in enumerate(frames):
#         try:
#             b0[i] = bond_grps.get_group(frame)['atom0'].astype(np.int64).values
#             b1[i] = bond_grps.get_group(frame)['atom1'].astype(np.int64).values
#         except Exception:
#             b0[i] = []
#             b1[i] = []
#     b0 = pd.Series(b0).to_json(orient='values')
#     b1 = pd.Series(b1).to_json(orient='values')
#     del uni.atom_two['frame']
#     return {'two_b0': b0, 'two_b1': b1}
#
#
# def frame_traits(uni):
#     """Get frame table traits."""
#     if not hasattr(uni, 'frame'): return {}
#     return {}
#
#
# class GUI(VBox):
#
#     def _close(self, b):
#         """Shut down all active widgets within the container."""
#         for widget in self.active_controls.values():
#             try: widget._close()
#             except: widget.close()
#
#         for widget in self.inactive_controls.values():
#             try: widget._close()
#             except: widget.close()
#
#         for scene in self.scenes: scene._close()
#         self.close()
#
#     def _handle_custom_msg(self, msg, callback):
#         """Custom message handler."""
#         print(msg)
#         # if msg['type'] == 'link_controls':
#         #     self._handle_image(msg['content'])
#         # elif msg['type'] == 'camera':
#         #     self._handle_camera(msg['content'])
#         # else: print("Custom msg not handled.\n"
#         #             "type of msg : {}\n"
#         #             "msg         : {}".format(msg['type'], msg['content']))
#
#     def _link_controls(self, b):
#         print('sending message')
#         self.send({'type': 'link_controls',
#                    'content': 'true'})
#         if (self.active_controls['camera']
#                 .active_controls['link'].icon) == 'link':
#             (self.active_controls['camera']
#                  .active_controls['link'].icon) = 'unlink'
#         else:
#             (self.active_controls['camera']
#                  .active_controls['link'].icon) = 'link'
#
#
#     def _init_gui(self, nframes=1, fields=None):
#         """Initialize generic GUI controls and register callbacks."""
#
#         # Default GUI controls to control the scene
#         self.inactive_controls = ODict()
#         self.active_controls = ODict(
#             close=Button(icon='trash', description=' Close', layout=gui_lo),
#             clear=Button(icon='bomb', description=' Clear', layout=gui_lo))
#
#         maxcam = 0
#         if len(self.scenes):
#             maxcam = len(self.scenes[0].cameras)
#         copts = ODict([
#             ('link', Button(icon='link', description=' Link')),
#             ('get', Button(icon='arrow-circle-down', description=' Save')),
#             ('set', IntSlider(description='Load', min=-1,
#                               max=maxcam - 1,
#                               value=-1, step=1))])
#         def _save_camera(b):
#             maxcam = 0
#             if len(self.scenes):
#                 maxcam = len(self.scenes[0].cameras)
#             for scene in self.scenes:
#                 scene.save_cam = scene.save_cam == False
#             (self.active_controls['camera']
#                  .active_controls['set'].max) = maxcam
#         copts['get'].on_click(_save_camera)
#         copts['link'].on_click(self._link_controls)
#         for scene in self.scenes:
#             copts['set'].observe(scene._set_camera, names='value')
#         cfolder = Folder(Button(icon='camera', description=' Camera'), copts)
#         cfolder.activate()
#
#         self.active_controls['camera'] = cfolder
#         self.active_controls['saves'] = Button(
#             icon='save', description=' Save', layout=gui_lo)
#
#         def _clear(b):
#             for scene in self.scenes:
#                 scene.clear = scene.clear == False
#         def _saves(b):
#             for scene in self.scenes:
#                 scene.save = scene.save == False
#
#         self.active_controls['close'].on_click(self._close)
#         self.active_controls['clear'].on_click(_clear)
#         self.active_controls['saves'].on_click(_saves)
#
#         # Inactive GUI controls common for subclasses
#         fopts = gui_field_widgets(True, False)
#
#         def _iso(c):
#             for scene in self.scenes: scene.field_iso = c.new
#         def _nx(c):
#             for scene in self.scenes: scene.field_nx = c.new
#         def _ny(c):
#             for scene in self.scenes: scene.field_ny = c.new
#         def _nz(c):
#             for scene in self.scenes: scene.field_nz = c.new
#
#         fopts['iso'].observe(_iso, names='value')
#         fopts['nx'].observe(_nx, names='value')
#         fopts['ny'].observe(_ny, names='value')
#         fopts['nz'].observe(_nz, names='value')
#
#         self.inactive_controls['field'] = Folder(
#                 Button(description=' Fields', icon='cube'), fopts)
#
#         atoms = Button(description=' Fill', icon='adjust', layout=gui_lo)
#         axis = Button(description=' Axis', icon='arrows-alt', layout=gui_lo)
#         def _atom_3d(b):
#             for scene in self.scenes: scene.atom_3d = scene.atom_3d == False
#         def _axis(b):
#             for scene in self.scenes: scene.axis = scene.axis == False
#         atoms.on_click(_atom_3d)
#         axis.on_click(_axis)
#         self.active_controls['atom_3d'] = atoms
#         self.active_controls['axis'] = axis
#
#         playable = bool(nframes <= 1)
#         flims = dict(min=0, max=nframes-1, step=1, value=0, layout=gui_lo)
#         control = Button(description=' Animate', icon='play')
#         content = ODict([
#             ('playing', Play(disabled=playable, **flims)),
#             ('scn_frame', IntSlider(description='Frame', **flims))
#         ])
#         def _scn_frame(c):
#             for scene in self.scenes: scene.frame_idx = c.new
#         content['scn_frame'].observe(_scn_frame, names='value')
#         jslink((content['playing'], 'value'),
#                (content['scn_frame'], 'value'))
#         self.active_controls['frame'] = Folder(control, content)
#         self.active_controls['frame'].activate()
#
#
#         if fields is not None:
#             # Main field folder
#             folder = self.inactive_controls.pop('field')
#             fopts = Dropdown(options=fields, layout=gui_lo)
#             def _fopts(c):
#                 for scene in self.scenes: scene.field_idx = c.new
#             fopts.observe(_fopts, names='value')
#             # Make an isosurface folder
#             isos = Button(description=' Isosurfaces', icon='cube')
#             def _fshow(b):
#                 for scene in self.scenes:
#                     scene.field_show = scene.field_show == False
#             isos.on_click(_fshow)
#             # Move the isosurface button to the subfolder
#             iso = folder.inactive_controls.pop('iso')
#             isofolder = Folder(isos, ODict([
#                 ('fopts', fopts),
#                 ('iso', iso)
#             ]), layout=Layout(width="200px"))
#             isofolder.activate()
#             folder.insert(1, 'iso', isofolder, update=True)
#             # Make a contour folder
#             control = Button(description=' Contours', icon='dot-circle-o')
#             def _cshow(b):
#                 for scene in self.scenes:
#                     scene.cont_show = scene.cont_show == False
#             control.on_click(_cshow)
#             content = ODict([
#                 ('fopts', fopts),
#                 ('axis', Dropdown(options=['x', 'y', 'z'], value='z')),
#                 ('num', IntSlider(description='N', min=5, max=20,
#                                   value=10, step=1, layout=gui_lo,)),
#                 ('lim', IntRangeSlider(description="10**Limits", min=-8,
#                                        max=0, step=1, value=[-7, -1],)),
#                 ('val', FloatSlider(description="Value",
#                                     min=-5, max=5, value=0,)),
#             ])
#             def _cont_axis(c):
#                 for scene in self.scenes: scene.cont_axis = c.new
#             def _cont_num(c):
#                 for scene in self.scenes: scene.cont_num = c.new
#             def _cont_lim(c):
#                 for scene in self.scenes: scene.cont_lim = c.new
#             def _cont_val(c):
#                 for scene in self.scenes: scene.cont_val = c.new
#             content['axis'].observe(_cont_axis, names='value')
#             content['num'].observe(_cont_num, names='value')
#             content['lim'].observe(_cont_lim, names='value')
#             content['val'].observe(_cont_val, names='value')
#             contour = Folder(control, content, layout=Layout(width="200px"))
#             contour.activate()
#             folder.insert(2, 'contour', contour, update=True)
#             self.active_controls['field'] = folder
#
#
#     def __init__(self, *unis, **kwargs):
#         scenekwargs = kwargs.pop("scenekwargs", None)
#         scenes = []
#         flds = []
#         for uni in unis:
#             unargs = {}
#             fields = None
#             atomcolors, atomradii = {}, {}
#             scenekwargs = {} if scenekwargs is None else scenekwargs
#             if 'atomcolors' in scenekwargs:
#                 atomcolors = scenekwargs['atomcolors']
#             if 'atomradii' in scenekwargs:
#                 atomradii = scenekwargs['atomradii']
#             if hasattr(uni, 'atom'):
#                 unargs.update(atom_traits(uni.atom, atomcolors, atomradii))
#             if hasattr(uni, 'atom_two'):
#                 unargs.update(two_traits(uni))
#             if hasattr(uni, 'field'):
#                 unargs.update(field_traits(uni.field))
#                 fields = ['null'] + unargs['field_i'][0]
#                 if len(fields) > len(flds):
#                     flds = fields
#             if scenekwargs is not None: unargs.update(scenekwargs)
#             scenes.append(UniverseScene(**unargs))
#         nframes = max((uni.atom.nframes for uni in unis)) if len(unis) else 1
#         #nframes = max(*(uni.atom.nframes for uni in unis), 0)
#         self.scenes = scenes
#         self._init_gui(nframes=nframes, fields=flds)
#         children = [VBox(list(self.active_controls.values())),
#                          HBox(self.scenes)]
#         super(GUI, self).__init__(children=children,
#                                   **kwargs)
        # if len(self.scenes):
        #     self.scenes[0].save_cam = self.scenes[0].save_cam == False
        #     sleep(1.0)
        #     #_save_camera(True)
        #     for scene in self.scenes[1:]:
        #         scene._set_camera({'new': self.scenes[0].cameras[0]})

    #     def _save_camera(b):
    #         maxcam = 0
    #         if len(self.scenes):
    #             maxcam = len(self.scenes[0].cameras)
    #         for scene in self.scenes:
    #             scene.save_cam = scene.save_cam == False
    #         (self.active_controls['camera']
    #              .active_controls['set'].max) = maxcam
    # def _set_camera(self, c):
    #     if c.new == -1: return
    #     self.send({'type': 'camera',
    #                'content': self.cameras[c.new]})

#
# @register
# class UniverseScene(ExatomicScene):
#     """A scene for viewing quantum systems."""
#     _model_name = Unicode("UniverseSceneModel").tag(sync=True)
#     _view_name = Unicode("UniverseSceneView").tag(sync=True)
#     # Top level index
#     frame_idx = Int(0).tag(sync=True)
#     axis = Bool(False).tag(sync=True)
#     # Atom traits
#     atom_x = Unicode().tag(sync=True)
#     atom_y = Unicode().tag(sync=True)
#     atom_z = Unicode().tag(sync=True)
#     atom_l = Unicode().tag(sync=True)
#     atom_s = Unicode().tag(sync=True)
#     atom_r = Dict().tag(sync=True)
#     atom_c = Dict().tag(sync=True)
#     atom_3d = Bool(False).tag(sync=True)
#     # Two traits
#     two_b0 = Unicode().tag(sync=True)
#     two_b1 = Unicode().tag(sync=True)
#     # Field traits
#     field_i = List().tag(sync=True)
#     #field_v = Instance(NDArrayWidget).tag(sync=True, **widget_serialization)
#     #field_v = NDArray().tag(sync=True, **array_serialization)
#     field_v = Unicode().tag(sync=True)
#     field_p = Dict().tag(sync=True)
#     field_idx = Any().tag(sync=True)
#     field_iso = Float(0.03).tag(sync=True)
#     field_show = Bool(False).tag(sync=True)
#     cont_show = Bool(False).tag(sync=True)
#     cont_axis = Unicode("z").tag(sync=True)
#     cont_num = Int(10).tag(sync=True)
#     cont_lim = List([-8, -1]).tag(sync=True)
#     cont_val = Float(0.0).tag(sync=True)
#     # Frame traits
#
#
# #@register
# class UniverseWidget(ExatomicBox):
#     """:class:`~exatomic.container.Universe` viewing widget."""
#
#     def _init_gui(self, nframes=1, fields=None):
#
#         super(UniverseWidget, self)._init_gui(uni=True, test=False)
#
#         atoms = Button(description=' Fill', icon='adjust', layout=gui_lo)
#         axis = Button(description=' Axis', icon='arrows-alt', layout=gui_lo)
#         def _atom_3d(b): self.scene.atom_3d = self.scene.atom_3d == False
#         def _axis(b): self.scene.axis = self.scene.axis == False
#         atoms.on_click(_atom_3d)
#         axis.on_click(_axis)
#         self.active_controls['atom_3d'] = atoms
#         self.active_controls['axis'] = axis
#
#         playable = bool(nframes <= 1)
#         flims = dict(min=0, max=nframes-1, step=1, value=0, layout=gui_lo)
#         control = Button(description=' Animate', icon='play')
#         content = OrderedDict([
#             ('playing', Play(disabled=playable, **flims)),
#             ('scn_frame', IntSlider(description='Frame', **flims))
#         ])
#         def _scn_frame(c): self.scene.frame_idx = c.new
#         content['scn_frame'].observe(_scn_frame, names='value')
#         jslink((content['playing'], 'value'),
#                (content['scn_frame'], 'value'))
#         self.active_controls['frame'] = Folder(control, content)
#         self.active_controls['frame'].activate()
#
#
#         if fields is not None:
#             # Main field folder
#             folder = self.inactive_controls.pop('field')
#             fopts = Dropdown(options=fields, layout=gui_lo)
#             def _fopts(c): self.scene.field_idx = c.new
#             fopts.observe(_fopts, names='value')
#             # Make an isosurface folder
#             isos = Button(description=' Isosurfaces', icon='cube')
#             def _fshow(b):
#                 self.scene.field_show = self.scene.field_show == False
#             isos.on_click(_fshow)
#             # Move the isosurface button to the subfolder
#             iso = folder.inactive_controls.pop('iso')
#             isofolder = Folder(isos, OrderedDict([
#                 ('fopts', fopts),
#                 ('iso', iso)
#             ]), layout=Layout(width="200px"))
#             isofolder.activate()
#             folder.insert(1, 'iso', isofolder, update=True)
#             # Make a contour folder
#             control = Button(description=' Contours', icon='dot-circle-o')
#             def _cshow(b):
#                 self.scene.cont_show = self.scene.cont_show == False
#             control.on_click(_cshow)
#             content = OrderedDict([
#                 ('fopts', fopts),
#                 ('axis', Dropdown(options=['x', 'y', 'z'], value='z')),
#                 ('num', IntSlider(description='N', min=5, max=20,
#                                   value=10, step=1, layout=gui_lo,)),
#                 ('lim', IntRangeSlider(description="10**Limits", min=-8,
#                                        max=0, step=1, value=[-7, -1],)),
#                 ('val', FloatSlider(description="Value",
#                                     min=-5, max=5, value=0,)),
#             ])
#             def _cont_axis(c): self.scene.cont_axis = c.new
#             def _cont_num(c): self.scene.cont_num = c.new
#             def _cont_lim(c): self.scene.cont_lim = c.new
#             def _cont_val(c): self.scene.cont_val = c.new
#             content['axis'].observe(_cont_axis, names='value')
#             content['num'].observe(_cont_num, names='value')
#             content['lim'].observe(_cont_lim, names='value')
#             content['val'].observe(_cont_val, names='value')
#             contour = Folder(control, content, layout=Layout(width="200px"))
#             contour.activate()
#             folder.insert(2, 'contour', contour, update=True)
#             self.active_controls['field'] = folder
#
#     def __init__(self, uni, *args, **kwargs):
#         scenekwargs = kwargs.pop("scenekwargs", None)
#         unargs = {}
#         fields = None
#         atomcolors, atomradii = {}, {}
#         scenekwargs = {} if scenekwargs is None else scenekwargs
#         if 'atomcolors' in scenekwargs:
#             atomcolors = scenekwargs['atomcolors']
#         if 'atomradii' in scenekwargs:
#             atomradii = scenekwargs['atomradii']
#         if hasattr(uni, 'atom'):
#             unargs.update(atom_traits(uni.atom, atomcolors, atomradii))
#         if hasattr(uni, 'atom_two'):
#             unargs.update(two_traits(uni))
#         if hasattr(uni, 'field'):
#             unargs.update(field_traits(uni.field))
#             fields = ['null'] + unargs['field_i'][0]
#         if scenekwargs is not None: unargs.update(scenekwargs)
#         self.scene = UniverseScene(**unargs)
#         self._init_gui(nframes=uni.atom.nframes, fields=fields)
#         super(UniverseWidget, self).__init__(
#             *args, **kwargs)
#             # *args, scene=scene, **kwargs)
