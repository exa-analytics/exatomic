# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Lower Level Widgets
#########################
Lower level support widgets framework.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
#import numpy as np
from base64 import b64decode
from traitlets import (Bool, Int, Float, Unicode,
                       List, Any, Dict, link)
from ipywidgets import (
    Box, VBox, HBox, IntSlider, Text, ToggleButton,
    DOMWidget, Layout, Button, Dropdown, register)

from exatomic import __js_version__
#from .traits import uni_traits
#from .widget_utils import (_glo, _flo, _wlo, _hboxlo,
from .widget_utils import (_wlo, _hboxlo,
                           _vboxlo, _bboxlo, _ListDict,
                           Folder, GUIBox, gui_field_widgets)


@register
class ExatomicScene(DOMWidget):
    """
    A custom scene (three.js scene) used for visualization
    of an atomic universe.

    Custom parameters can be inspected using ``vars``. Parameters
    include field dimensions, the universe itself, field colors,
    widths, heights for the widget, etc.
    """
    _model_module_version = Unicode(__js_version__).tag(sync=True)
    _view_module_version = Unicode(__js_version__).tag(sync=True)
    _view_module = Unicode('exatomic').tag(sync=True)
    _model_module = Unicode('exatomic').tag(sync=True)
    _model_name = Unicode('ExatomicSceneModel').tag(sync=True)
    _view_name = Unicode('ExatomicSceneView').tag(sync=True)
    # Base controls and GUI
    savedir = Unicode(os.getcwd()).tag(sync=True)
    imgname = Unicode().tag(sync=True)
    index = Int().tag(sync=True) # doesn't need sync
    cameras = List(trait=Dict()).tag(sync=True)
    save_cam = Bool(False).tag(sync=True)
    clear = Bool(False).tag(sync=True)
    save = Bool(False).tag(sync=True)
    w = Int(200).tag(sync=True)
    h = Int(200).tag(sync=True)
    field_pos = Unicode('#003399').tag(sync=True)
    field_neg = Unicode('#FF9900').tag(sync=True)
    field_iso = Float(2.0).tag(sync=True)
    field_o = Float(1.0).tag(sync=True)
    field = Unicode('null').tag(sync=True)
    field_kind = Unicode('').tag(sync=True)
    field_ml = Unicode('0').tag(sync=True)
    # Test containers
    # test = Bool(False).tag(sync=True) # doesn't need sync
    uni = Bool(False).tag(sync=True)  # doesn't need sync
    field_nx = Int(31).tag(sync=True)
    field_ny = Int(31).tag(sync=True)
    field_nz = Int(31).tag(sync=True)
    field_ox = Float(-3.0).tag(sync=True)
    field_oy = Float(-3.0).tag(sync=True)
    field_oz = Float(-3.0).tag(sync=True)
    field_fx = Float(3.0).tag(sync=True)
    field_fy = Float(3.0).tag(sync=True)
    field_fz = Float(3.0).tag(sync=True)
    geom = Bool(True).tag(sync=True)

    def _handle_custom_msg(self, msg, callback):
        """Custom message handler."""
        if msg['type'] == 'image':
            self._save_image(msg['content'])
        elif msg['type'] == 'camera':
            self._save_camera(msg['content'])
        else: print('Custom msg not handled.\n'
                    'type of msg : {}\n'
                    'msg         : {}'.format(msg['type'],
                                              msg['content']))

    def _save_camera(self, content):
        """Cache a save state of the current camera."""
        self.cameras.append(content)

    def _save_image(self, content):
        """Save a PNG of the scene."""

        savedir = self.savedir
        if savedir != os.getcwd():
            if not os.path.isdir(savedir):
                raise Exception('Must supply a valid directory.')
        if not savedir.endswith(os.sep):
            savedir += os.sep

        nxt = 0
        fmt = '{:06d}.png'.format
        fname = self.imgname
        if fname == 'name' or not fname:
            fname = fmt(nxt)
        while os.path.isfile(os.sep.join([savedir, fname])):
            nxt += 1
            fname = fmt(nxt)

        if self.index:
            ext = '-{}.png'.format(self.index)
            if fname.endswith('.png'):
                fname = fname.replace('.png', ext)
            else:
                fname = fname + ext
        if not fname.endswith('.png'):
            fname += '.png'

        repl = 'data:image/png;base64,'
        with open(os.sep.join([savedir, fname]), 'wb') as f:
            f.write(b64decode(content.replace(repl, '')))

    def _set_camera(self, c):
        """Ship the camera to JS to set a cached camera."""
        if c.new == -1: return
        try:
            self.send({'type': 'camera', 'content': self.cameras[c.new]})
        except IndexError:
            pass

    def _close(self):
        """Close the three.js objects and then close."""
        self.send({'type': 'close'})
        self.close()

    def __init__(self, *args, **kwargs):
        lo = kwargs.pop('layout', None)
        if lo is None:
            height = kwargs.pop('height', 'auto')
            min_height = kwargs.pop('min_height', '400px')
            min_width = kwargs.pop('min_width', '300px')
            flex = kwargs.pop('flex', '1 1 auto')
            lo = Layout(height=height, min_height=min_height,
                        flex=flex, min_width=min_width)
        super(DOMWidget, self).__init__(
            *args, layout=lo, **kwargs)


@register
class TensorScene(ExatomicScene):
    """Visualize a cartesian tensor."""

    _model_name = Unicode('TensorSceneModel').tag(sync=True)
    _view_name = Unicode('TensorSceneView').tag(sync=True)
    field = Unicode('null').tag(sync=True)
    geom = Bool(True).tag(sync=True)
    txx = Float(1.).tag(sync=True)
    txy = Float(0.).tag(sync=True)
    txz = Float(0.).tag(sync=True)
    tyx = Float(0.).tag(sync=True)
    tyy = Float(1.).tag(sync=True)
    tyz = Float(0.).tag(sync=True)
    tzx = Float(0.).tag(sync=True)
    tzy = Float(0.).tag(sync=True)
    tzz = Float(1.).tag(sync=True)
    scale = Float(1.).tag(sync=True)
    tdx = Int(0).tag(sync=True)
    tens = Bool(False).tag(sync=True)


@register
class UniverseScene(ExatomicScene):
    """A scene for viewing quantum systems."""
    _model_name = Unicode('UniverseSceneModel').tag(sync=True)
    _view_name = Unicode('UniverseSceneView').tag(sync=True)
    # Top level index
    frame_idx = Int(0).tag(sync=True)
    axis = Bool(False).tag(sync=True)
    # Atom traits
    atom_x = Unicode().tag(sync=True)
    atom_y = Unicode().tag(sync=True)
    atom_z = Unicode().tag(sync=True)
    atom_l = Dict().tag(sync=True)
    atom_s = Unicode().tag(sync=True)
#    atom_r = Dict().tag(sync=True)
    atom_vr = Dict().tag(sync=True)
    atom_cr = Dict().tag(sync=True)
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
    field_show = Bool(False).tag(sync=True)
    cont_show = Bool(False).tag(sync=True)
    cont_axis = Unicode('z').tag(sync=True)
    cont_num = Int(10).tag(sync=True)
    cont_lim = List([-8, -1]).tag(sync=True)
    cont_val = Float(0.0).tag(sync=True)
    # Frame traits
    frame__a = Float(0.0).tag(sync=True)
    # Tensor traits
    tens = Bool(False).tag(sync=True)
    tensor_d = Dict().tag(sync=True)
    scale = Float(1.).tag(sync=True)
    tidx = Int(0).tag(sync=True)
    # View traits
    fill_idx = Int(0).tag(sync=True)
    bond_r = Float(-1).tag(sync=True)
    selected = Dict().tag(sync=True)
    clear_selected = Bool(False).tag(sync=True)

    # This block works to print out changes from javascript
    #@observe('selected')
    #def _observe_selected(self, change):
    #    print(change['old'])
    #    print(change['new'])


@register
class ExatomicBox(Box):
    """Base class for containers of a GUI and scene."""
    _model_module_version = Unicode(__js_version__).tag(sync=True)
    _view_module_version = Unicode(__js_version__).tag(sync=True)
    _model_module = Unicode('exatomic').tag(sync=True)
    _view_module = Unicode('exatomic').tag(sync=True)
    _model_name = Unicode('ExatomicBoxModel').tag(sync=True)
    _view_name = Unicode('ExatomicBoxView').tag(sync=True)
    active_scene_indices = List().tag(sync=True)
    linked = Bool(False).tag(sync=True)

    def _update_active(self, b):
        """Control which scenes are controlled by the GUI."""
        items = list(self._controls['active']._controls.items())[1:]
        self.active_scene_indices = [i for i, (key, obj) in enumerate(items)
                                     if obj.value]

    def _close(self, b):
        """Shut down all active widgets within the container."""
        for widget in self._controls.values():
            try: widget._close()
            except AttributeError: widget.close()
        for widget in self.scenes:
            try: widget._close()
            except AttributeError: widget.close()
        self.close()

    def _get(self, active=True, keys=False):
        """Get all the active GUI objects."""
        mit = self._controls.values()
        return [obj for obj in mit if obj.active]

    def _active_folder(self):
        """Folder that houses the controls for active scenes."""
        active = Button(icon='bars', description=' Active Scenes')
        opts = _ListDict([
            (str(i), ToggleButton(description=str(i), value=True))
            for i, scn in enumerate(self.scenes)])
        for obj in opts.values():
            obj.observe(self._update_active, names='value')
        return Folder(active, opts)

    def _save_folder(self):
        """Folder that houses the controls to save images."""
        saves = Button(icon='save', description=' Image')
        saveopts = _ListDict([
            ('dir', Text(value=os.getcwd())),
            ('name', Text(value='name')),
            ('save', Button(icon='download', description=' Save'))
        ])
        for scn in self.scenes:
            link((saveopts['dir'], 'value'), (scn, 'savedir'))
            link((saveopts['name'], 'value'), (scn, 'imgname'))
        def _saves(b):
            for scn in self.active():
                scn.save = not scn.save
        saveopts['save'].on_click(_saves)
        return Folder(saves, saveopts)

    def _camera_folder(self):
        """Folder that houses controls for caching and setting cameras."""
        ncams = max((len(scn.cameras) for scn
                    in self.scenes)) if self.scenes else 0
        camera = Button(icon='camera', description=' Camera')
        camopts = _ListDict([
             ('get', Button(icon='arrow-circle-down', description=' Save')),
             ('set', IntSlider(description='Load', min=-1,
                               max=ncams-1, value=-1, step=1))])

        def _save_cam(b):
            for scn in self.active():
                scn.save_cam = not scn.save_cam
                btn = self._controls['camera']._controls['set']
                btn.max = len(scn.cameras)

        camopts['get'].on_click(_save_cam)
        for scn in self.scenes:
            camopts['set'].observe(scn._set_camera, names='value')
        if len(self.scenes) > 1:
            camopts.insert(0, 'link', Button(icon='link', description=' Link'))
            def _link(b):
                self.linked = not self.linked
                btn = self._controls['camera']._controls['link']
                if self.linked:
                    btn.icon = 'unlink'
                    btn.description = ' Unlink'
                else:
                    btn.icon = 'link'
                    btn.description = ' Link'
            camopts['link'].on_click(_link)
        return Folder(camera, camopts)

    def _field_folder(self, **kwargs):
        """Folder that houses controls for viewing scalar fields."""
        uni = kwargs.pop('uni', False)
        #test = kwargs.pop('test', True)
        fdict = gui_field_widgets(uni)#, test)

        def _iso(c):
            for scn in self.active():
                scn.field_iso = c.new
        def _alpha(c):
            for scn in self.active():
                scn.field_o = c.new
        def _nx(c):
            for scn in self.active():
                scn.field_nx = c.new
        def _ny(c):
            for scn in self.active():
                scn.field_ny = c.new
        def _nz(c):
            for scn in self.active():
                scn.field_nz = c.new

        fdict['iso'].observe(_iso, names='value')
        fdict['alpha'].observe(_alpha, names='value')
        fdict['nx'].observe(_nx, names='value')
        fdict['ny'].observe(_ny, names='value')
        fdict['nz'].observe(_nz, names='value')
        field = Button(description=' Fields', icon='cube')
        folder = Folder(field, fdict)
        return folder

    def _init_gui(self, **kwargs):
        """Initialize generic GUI controls and observe callbacks."""
        mainopts = _ListDict([
            ('close', Button(icon='trash', description=' Close', layout=_wlo)),
            ('clear', Button(icon='bomb', description=' Clear', layout=_wlo))])
        mainopts['close'].on_click(self._close)

        if self.scenes:
            def _clear(b):
                for scn in self.active():
                    scn.clear = not scn.clear
            mainopts['close'].on_click(self._close)
            mainopts['clear'].on_click(_clear)
        mainopts.update([('active', self._active_folder()),
                         ('saves', self._save_folder()),
                         ('camera', self._camera_folder())])
        return mainopts

    def active(self):
        return [self.scenes[idx] for idx in self.active_scene_indices]

    def __init__(self, *objs, **kwargs):
        objs = (1,) if not objs else objs
        scenekwargs = kwargs.pop('scenekwargs', {})
        #test = kwargs.pop('test', False)
        uni = kwargs.pop('uni', False)
        typ = kwargs.pop('typ', ExatomicScene)
        mh = kwargs.pop('min_height', None)
        mw = kwargs.pop('min_width', None)
        nframes = kwargs.pop('nframes', None)
        fields = kwargs.pop('fields', None)
        tensors = kwargs.pop('tensors', None)
        self.scenes, scenes = _scene_grid(objs, mh, mw,# test,
                                          uni, typ, scenekwargs)
        self._controls = self._init_gui(nframes=nframes,
                                        fields=fields,
                                        tensors=tensors,
                                        #test=test,
                                        uni=uni)
        for _, obj in self._controls.items():
            if not hasattr(obj, 'active'): obj.active = True
        _ = kwargs.pop('layout', None)
        gui = GUIBox(self._get())
        children = [gui, scenes] if self.scenes else [gui]
        super(ExatomicBox, self).__init__(
                children, layout=_bboxlo,
                active_scene_indices=list(range(len(self.scenes))),
                **kwargs)


#def _scene_grid(objs, mh, mw, test, uni, typ, scenekwargs):
def _scene_grid(objs, mh, mw, uni, typ, scenekwargs):
    """Auxiliary function to lay out multiple scenes."""
    n = objs[0] if isinstance(objs[0], int) else len(objs)
    if n > 9: raise NotImplementedError('Too many scenes')
    if mh is None:
        if n < 2: mh = '500px'
        elif n < 3: mh = '400px'
        elif n < 5: mh = '300px'
        elif n < 7: mh = '250px'
        else: mh = '200px'
    if n < 5: mod = 2
    else: mod = 3
    kwargs = {'min_height': mh, 'min_width': mw, 'uni': uni}
    kwargs.update(scenekwargs)
    flatscns, scenes = [], []
    for i in range(n):
        kwargs['index'] = i
        if not i % mod: scenes.append([])
        try:
            if isinstance(objs[i], DOMWidget):
                obj = objs[i]
            elif isinstance(objs[i], dict):
                objs[i].update(kwargs)
                obj = typ(**objs[i])
            else:
                obj = typ(**kwargs)
        except IndexError:
            obj = typ(**kwargs)
        scenes[-1].append(obj)
        flatscns.append(obj)
    return flatscns, VBox([HBox(scns, layout=_hboxlo)
                           for scns in scenes], layout=_vboxlo)
