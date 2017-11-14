# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Universe Notebook Widget
#########################
"""

from traitlets import Bool, Unicode, Float, Dict, Any, List, Int
from ipywidgets import (Button, Dropdown, jslink, register, VBox, HBox,
                        IntSlider, IntRangeSlider, FloatSlider, Play)

from .widget_base import ExatomicScene, ExatomicBox
from .widget_utils import _wlo, _hboxlo, _vboxlo, _ListDict, _scene_grid, Folder
from .traits import ( #atom_traits, field_traits, two_traits, frame_traits,
uni_traits)


class TestContainer(ExatomicBox):


    def _field_folder(self):
        folder = super(TestContainer, self)._field_folder()
        fopts = ['null', 'Sphere', 'Torus', 'Ellipsoid']
        fopts = Dropdown(options=fopts)
        fopts.active = True
        fopts.disabled = False
        def _field(c):
            # self._update_active()
            for idx in self.active_scene_indices:
                self.scenes[idx].field = c.new
        fopts.observe(_field, names='value')
        folder.insert(1, 'options', fopts)
        return folder


    def _init_gui(self, **kwargs):
        mainopts = super(TestContainer, self)._init_gui()
        geom = Button(icon="gear", description=" Mesh", layout=_wlo)
        def _geom(b):
            # self._update_active()
            for idx in self.active_scene_indices:
                self.scenes[idx].geom = self.scenes[idx].geom == False
        geom.on_click(_geom)
        mainopts.update([('geom', geom),
                         ('field', self._field_folder())])
        return mainopts


    def __init__(self, *scenes, **kwargs):
        self.uni = False
        self.test = True
        super(TestContainer, self).__init__(*scenes, **kwargs)



class TestUniverse(ExatomicBox):

    def _update_active(self, b):
        super(TestUniverse, self)._update_active(b)
        scns = [self.scenes[idx] for idx in self.active_scene_indices]
        if not scns: return
        flds = [scn.field for scn in scns]
        fks = [scn.field_kind for scn in scns]
        fmls = [scn.field_ml for scn in scns]
        folder = self._controls['field']
        fopts = folder['fopts'].options
        fld = flds[0]
        fk = fks[0]
        fml = fmls[0]

        if not len(set(flds)) == 1:
            for scn in scns: scn.field = fld
        if not len(set(fks)) == 1:
            for scn in scns: scn.field_kind = fk
        if not len(set(fmls)) == 1:
            for scn in scns: scn.field_ml = fml

        folder[fld].value = fk
        folder.activate(fld, enable=True)
        folder.deactivate(*[f for f in fopts if f != fld])
        if fld == 'SolidHarmonic':
            ofks = [str(i) for i in range(8) if str(i) != fk]
            folder.activate(fk, enable=True)
            folder.deactivate(*ofks)
        folder._set_gui()


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
            for idx in self.active_scene_indices:
                self.scenes[idx].field = c.new
            fk = uni_field_lists[c.new][0]
            for idx in self.active_scene_indices:
                self.scenes[idx].field_kind = fk
            self._update_active(None)

        def _field_kind(c):
            for idx in self.active_scene_indices:
                self.scenes[idx].field_kind = c.new
                if self.scenes[idx].field == 'SolidHarmonic':
                    self.scenes[idx].field_ml = folder[c.new].options[0]
            self._update_active(None)

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


    def _init_gui(self, **kwargs):
        mainopts = super(TestUniverse, self)._init_gui()
        mainopts.update([('field', self._field_folder())])
        return mainopts


    def __init__(self, *scenes, **kwargs):
        self.uni = True
        self.test = True
        super(TestUniverse, self).__init__(*scenes, **kwargs)




@register
class UniverseWidget(ExatomicBox):
    """:class:`~exatomic.container.Universe` viewing widget."""


    def _frame_folder(self, nframes):
        playable = bool(nframes <= 1)
        flims = dict(min=0, max=nframes-1, step=1, value=0)
        control = Button(description=' Animate', icon='play')
        content = _ListDict([
            ('playing', Play(disabled=playable, **flims)),
            ('scn_frame', IntSlider(description='Frame', **flims))])
        def _scn_frame(c):
            # self._update_active()
            for idx in self.active_scene_indices:
                self.scenes[idx].frame_idx = c.new
        content['scn_frame'].observe(_scn_frame, names='value')
        jslink((content['playing'], 'value'),
               (content['scn_frame'], 'value'))
        folder = Folder(control, content)
        #folder.deactivate('playing', 'scn_frame')
        return folder

    def _field_folder(self, fields):
        # Main field folder
        folder = super(UniverseWidget, self)._field_folder()
        folder.deactivate('nx', 'ny', 'nz')
        fopts = Dropdown(options=fields)
        def _fopts(c):
            # self._update_active()
            for idx in self.active_scene_indices:
                self.scenes[idx].field_idx = c.new
        fopts.observe(_fopts, names='value')
        folder['fopts'] = fopts
        return folder


    def _iso_folder(self, folder):
        # Isosurface folder
        isos = Button(description=' Isosurfaces', icon='cube')
        def _fshow(b):
            # self._update_active()
            for idx in self.active_scene_indices:
                self.scenes[idx].field_show = self.scenes[idx].field_show == False
        isos.on_click(_fshow)
        # Move some buttons to the subfolder
        isofolder = Folder(isos, _ListDict([
            ('fopts', folder['fopts']),
            ('alpha', folder.pop('alpha')),
            ('iso', folder.pop('iso'))]))#, level=1)
        # isofolder.activate()
        isofolder.move_to_end('alpha', 'iso')
        folder.insert(1, 'iso', isofolder, active=True)


    def _contour_folder(self, folder):
            # Make a contour folder
            control = Button(description=' Contours', icon='dot-circle-o')
            def _cshow(b):
                # self._update_active()
                for idx in self.active_scene_indices:
                    self.scenes[idx].cont_show = self.scenes[idx].cont_show == False
            control.on_click(_cshow)
            content = _ListDict([
                ('fopts', folder['fopts']),
                ('axis', Dropdown(options=['x', 'y', 'z'], value='z')),
                ('num', IntSlider(description='N', min=5, max=20,
                                  value=10, step=1)),
                ('lim', IntRangeSlider(description="10**Limits", min=-8,
                                       max=0, step=1, value=[-7, -1])),
                ('val', FloatSlider(description="Value",
                                    min=-5, max=5, value=0))])
            def _cont_axis(c):
                # self._update_active()
                for idx in self.active_scene_indices:
                    self.scenes[idx].cont_axis = c.new
            def _cont_num(c):
                # self._update_active()
                for idx in self.active_scene_indices:
                    self.scenes[idx].cont_num = c.new
            def _cont_lim(c):
                # self._update_active()
                for idx in self.active_scene_indices:
                    self.scenes[idx].cont_lim = c.new
            def _cont_val(c):
                # self._update_active()
                for idx in self.active_scene_indices:
                    self.scenes[idx].cont_val = c.new
            content['axis'].observe(_cont_axis, names='value')
            content['num'].observe(_cont_num, names='value')
            content['lim'].observe(_cont_lim, names='value')
            content['val'].observe(_cont_val, names='value')
            contour = Folder(control, content)#, level=1)
            folder.insert(2, 'contour', contour, active=True, update=True)



    def _init_gui(self, nframes=1, fields=None):

        mainopts = super(UniverseWidget, self)._init_gui()

        atoms = Button(description=' Fill', icon='adjust', layout=_wlo)
        axis = Button(description=' Axis', icon='arrows-alt', layout=_wlo)
        def _atom_3d(b):
            # self._update_active()
            for idx in self.active_scene_indices:
                self.scenes[idx].atom_3d = self.scenes[idx].atom_3d == False
        def _axis(b):
            # self._update_active()
            for idx in self.active_scene_indices:
                self.scenes[idx].axis = self.scenes[idx].axis == False
        atoms.on_click(_atom_3d)
        axis.on_click(_axis)
        atoms.active = True
        atoms.disabled = False
        axis.active = True
        atoms.disabled = False

        mainopts.update([('atom_3d', atoms), ('axis', axis),
                         ('frame', self._frame_folder(nframes))])
        if fields is not None:
            folder = self._field_folder(fields)
            self._iso_folder(folder)
            self._contour_folder(folder)
            folder.pop('fopts')

            mainopts.update([('field', folder)])
        return mainopts


    def __init__(self, *unis, **kwargs):
        self.uni = True
        self.test = False
        scenekwargs = kwargs.pop("scenekwargs", {})
        scenekwargs.update({'uni': self.uni, 'test': self.test})
        atomcolors = scenekwargs.get('atomcolors', None)
        atomradii = scenekwargs.get('atomradii', None)
        fields, masterkwargs = [], []
        for uni in unis:
            unargs, flds = uni_traits(uni)
            fields = flds if len(flds) > len(fields) else fields
            unargs.update(scenekwargs)
            masterkwargs.append(unargs)
        nframes = max((uni.atom.nframes
                      for uni in unis)) if len(unis) else 1
        super(UniverseWidget, self).__init__(
            *masterkwargs, nframes=nframes, fields=fields, **kwargs)
