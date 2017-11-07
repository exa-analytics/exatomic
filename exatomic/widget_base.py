# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Universe Notebook Widget
#########################
"""
import os
from base64 import b64decode
from traitlets import (Bool, Int, Float, Unicode,
                       List, Any, Dict, Instance, link)
                       # Unused == Any, Instance

from ipywidgets import (
    Box, VBox, HBox, FloatSlider, IntSlider, Play, Text,
    IntRangeSlider, DOMWidget, Layout, Button, Dropdown,
    register, jslink
    # Unused = HBox, FloatSlider,
)
from exatomic import __js_version__
from .widget_utils import (_glo, _flo, _wlo, _hboxlo, _vboxlo,
                           _ListDict, _scene_grid,
                           Folder, GUIBox, gui_field_widgets)


@register
class ExatomicScene(DOMWidget):


    _model_module_version = Unicode(__js_version__).tag(sync=True)
    _model_module_version = Unicode(__js_version__).tag(sync=True)
    _view_module = Unicode("jupyter-exatomic").tag(sync=True)
    _model_module = Unicode("jupyter-exatomic").tag(sync=True)
    _model_name = Unicode("ExatomicSceneModel").tag(sync=True)
    _view_name = Unicode("ExatomicSceneView").tag(sync=True)
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
    field_pos = Unicode("#003399").tag(sync=True)
    field_neg = Unicode("#FF9900").tag(sync=True)
    field_iso = Float(2.0).tag(sync=True)
    field_o = Float(1.0).tag(sync=True)
    field = Unicode("null").tag(sync=True)
    field_kind = Unicode("").tag(sync=True)
    field_ml = Unicode("0").tag(sync=True)
    # Test containers
    test = Bool(False).tag(sync=True) # doesn't need sync
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
            self._handle_camera(msg['content'])
        else: print("Custom msg not handled.\n"
                    "type of msg : {}\n"
                    "msg         : {}".format(msg['type'],
                                              msg['content']))


    def _handle_camera(self, content):
        self.cameras.append(content)


    def _save_image(self, content):
        """Save a PNG of the scene."""
        savedir = self.savedir
        if savedir != os.getcwd():
            if not os.path.isdir(savedir):
                print('Must supply a valid directory.')
                return
        if not savedir.endswith(os.sep):
            savedir += os.sep
        nxt = 0
        fmt = '{:06d}.png'.format
        fname = self.imgname
        if fname == 'name' or not fname: fname = fmt(nxt)
        # fname = self.imgname if self.imgname else fmt(nxt)
        while os.path.isfile(os.sep.join([savedir, fname])):
            nxt += 1
            fname = fmt(nxt)
        if self.index:
            if fname.endswith('.png'):
                fname = fname.replace('.png', '-{}.png'.format(self.index))
            else:
                fname = fname + '{}.png'.format(self.index)
        if not fname.endswith('.png'):
            fname += '.png'
        repl = 'data:image/png;base64,'
        with open(os.sep.join([savedir, fname]), 'wb') as f:
            f.write(b64decode(content.replace(repl, '')))


    def _set_camera(self, c):
        if c.new == -1: return
        self.send({'type': 'camera', 'content': self.cameras[c.new]})


    def _close(self):
        self.send({'type': 'close'})
        self.close()


    def __init__(self, *args, **kwargs):
        lo = kwargs.pop('layout', None)
        if lo is None:
            height = kwargs.pop('height', '100%')
            min_height = kwargs.pop('min_height', '400px')
            min_width = kwargs.pop('min_width', '400px')
            flex = kwargs.pop('flex', '1 1 auto')
            lo = Layout(height=height, min_height=min_height,
                        flex=flex, min_width=min_width)
            print(lo)
        super(DOMWidget, self).__init__(
            *args, layout=lo, **kwargs)




@register
class ExatomicBox(Box):
    """Base class for containers of a GUI and scene."""

    _model_module_version = Unicode(__js_version__).tag(sync=True)
    _model_module_version = Unicode(__js_version__).tag(sync=True)
    _model_module = Unicode("jupyter-exatomic").tag(sync=True)
    _view_module = Unicode("jupyter-exatomic").tag(sync=True)
    _model_name = Unicode("ExatomicBoxModel").tag(sync=True)
    _view_name = Unicode("ExatomicBoxView").tag(sync=True)
    active_scene_indices = List().tag(sync=True)
    linked = Bool(False).tag(sync=True)

    def _close(self, b):
        """Shut down all active widgets within the container."""

        for widget in self._controls.values():
            try: widget._close()
            except: widget.close()
        for widget in self.scenes:
            try: widget._close()
            except: widget.close()
        self.close()

    def _get(self, active=True, keys=False):
        mit = self._controls.values()
        return [obj for obj in mit if obj.active]

    def _save_folder(self):
        saves = Button(icon='save', description=' Image')
        saveopts = _ListDict([
            ('dir', Text(value='directory')),
            ('name', Text(value='name')),
            ('save', Button(icon='download', description=' Save'))
        ])
        for scn in self.scenes:
            link((saveopts['dir'], 'value'), (scn, 'savedir'))
            link((saveopts['name'], 'value'), (scn, 'imgname'))
        def _saves(b):
            for idx in self.active_scene_indices:
                self.scenes[idx].save = self.scenes[idx].save == False
        saveopts['save'].on_click(_saves)
        return Folder(saves, saveopts)

    def _camera_folder(self):
        ncams = max((len(scn.cameras) for scn
                    in self.scenes)) if self.scenes else 0
        camera = Button(icon='camera', description=' Camera')
        camopts = _ListDict([
             ('get', Button(icon='arrow-circle-down', description=' Save')),
             ('set', IntSlider(description='Load', min=-1,
                               max=ncams-1, value=-1, step=1))])
        def _save_cam(b):
            for idx in self.active_scene_indices:
                scn = self.scenes[idx]
                scn.save_cam = scn.save_cam == False
                btn = self._controls['camera']._controls['set']
                btn.max = len(scn.cameras)
        camopts['get'].on_click(_save_cam)
        for scn in self.scenes:
            camopts['set'].observe(scn._set_camera, names='value')
        if len(self.scenes) > 1:
            camopts.insert(0, 'link', Button(icon='link', description=' Link'))
            def _link(b):
                self.linked = self.linked == False
                btn = self._controls['camera']._controls['link']
                if self.linked:
                    btn.icon = 'unlink'
                    btn.description = ' Unlink'
                else:
                    btn.icon = 'link'
                    btn.description = ' Link'
            camopts['link'].on_click(_link)
        return Folder(camera, camopts)

    def _field_folder(self):
        fdict = gui_field_widgets(self.uni, self.test)
        def _iso(c):
            for idx in self.active_scene_indices:
                self.scenes[idx].field_iso = c.new
        def _alpha(c):
            for idx in self.active_scene_indices:
                self.scenes[idx].field_o = c.new
        def _nx(c):
            for idx in self.active_scene_indices:
                self.scenes[idx].field_nx = c.new
        def _ny(c):
            for idx in self.active_scene_indices:
                self.scenes[idx].field_ny = c.new
        def _nz(c):
            for idx in self.active_scene_indices:
                self.scenes[idx].field_nz = c.new
        fdict['iso'].observe(_iso, names='value')
        fdict['alpha'].observe(_alpha, names='value')
        fdict['nx'].observe(_nx, names='value')
        fdict['ny'].observe(_ny, names='value')
        fdict['nz'].observe(_nz, names='value')

        field = Button(description=' Fields', icon='cube')
        folder = Folder(field, fdict)

        return folder

    def _init_gui(self):
        """Initialize generic GUI controls and register callbacks."""

        mainopts = _ListDict([
            ('close', Button(icon='trash', description=' Close', layout=_wlo)),
            ('clear', Button(icon='bomb', description=' Clear', layout=_wlo))])
        mainopts['close'].on_click(self._close)
        if self.scenes:
            def _clear(b):
                for idx in self.active_scene_indices:
                    self.scenes[idx].clear = self.scenes[idx].clear == False
            mainopts['close'].on_click(self._close)
            mainopts['clear'].on_click(_clear)

        mainopts.update([('saves', self._save_folder())])
        mainopts.update([('camera', self._camera_folder())])
        return mainopts


    def __init__(self, scenes, **kwargs):
        if not hasattr(self, 'uni'):
            self.uni = kwargs.pop('uni', False)
        if not hasattr(self, 'test'):
            self.test = kwargs.pop('test', True)
        if isinstance(scenes, DOMWidget):
            scenes.uni = self.uni
            scenes.test = self.test
            self.scenes = [scenes]
            scenes = VBox([HBox([scenes], layout=_hboxlo)], layout=_vboxlo)
        else:
            mh = kwargs.pop('min_height', None)
            mw = kwargs.pop('min_width', None)
            unis, grid = _scene_grid(scenes, min_height=mh, min_width=mw)
            scenes = VBox([HBox([ExatomicScene(**scn)
                           for scn in row], layout=_hboxlo)
                           for row in grid], layout=_vboxlo)
            self.scenes = []
            for scn in scenes.children:
                for sub in scn.children:
                    sub.uni = self.uni
                    sub.test = self.test
                    self.scenes.append(sub)
        self._controls = self._init_gui()
        for key, obj in self._controls.items():
            if not hasattr(obj, 'active'): obj.active = True
        self.active_scene_indices = list(range(len(self.scenes)))
        lo = kwargs.pop('layout', None)
        gui = GUIBox(self._get())
        if self.scenes:
            super(ExatomicBox, self).__init__([gui, scenes], **kwargs)
        else:
            super(ExatomicBox, self).__init__([gui], **kwargs)
