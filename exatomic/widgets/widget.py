# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Universe Notebook Widget
#########################
"""

from traitlets import Unicode, link
from ipywidgets import (Button, Dropdown, jslink, register, VBox, HBox,
                        IntSlider, IntRangeSlider, FloatSlider, Play,
                        FloatText, Layout)

from .widget_base import (ExatomicScene, UniverseScene,
                          TensorScene, ExatomicBox)
from .widget_utils import _wlo, _ListDict, Folder
from .traits import uni_traits



class TestContainer(ExatomicBox):
    """A proof-of-concept mixing GUI controls with a three.js scene."""

    def _field_folder(self, **kwargs):
        """Folder that houses field GUI controls."""

        folder = super(TestContainer, self)._field_folder(**kwargs)
        fopts = Dropdown(options=['null', 'Sphere', 'Torus', 'Ellipsoid'])
        fopts.active = True
        fopts.disabled = False

        def _field(c):
            for idx in self.active_scene_indices:
                self.scenes[idx].field = c.new

        fopts.observe(_field, names='value')
        folder.insert(1, 'options', fopts)

        return folder


    def _init_gui(self, **kwargs):
        """Initialize generic GUI controls and observe callbacks."""

        mainopts = super(TestContainer, self)._init_gui()

        geom = Button(icon='gear', description=' Mesh', layout=_wlo)
        def _geom(b):
            for idx in self.active_scene_indices:
                self.scenes[idx].geom = not self.scenes[idx].geom
        geom.on_click(_geom)

        mainopts.update([('geom', geom),
                         ('field', self._field_folder(**kwargs))])
        return mainopts


    def __init__(self, *scenes, **kwargs):
        super(TestContainer, self).__init__(*scenes,
                                            uni=False,
                                            test=True,
                                            typ=ExatomicScene,
                                            **kwargs)





@register
class TensorContainer(ExatomicBox):
    """A simple container to implement cartesian tensor visualization."""

    _model_name = Unicode('TensorContainerModel').tag(sync=True)
    _view_name = Unicode('TensorContainerView').tag(sync=True)

    def _init_gui(self, **kwargs):
        """Initialize generic GUI controls and observe callbacks."""

        mainopts = super(TensorContainer, self)._init_gui(**kwargs)

        geom = Button(icon='cubes', description=' Mesh', layout=_wlo)
        alo = Layout(width='74px')
        rlo = Layout(width='234px')

        xs = [FloatText(value=1., layout=alo),
              FloatText(value=0., layout=alo),
              FloatText(value=0., layout=alo)]
        ys = [FloatText(value=0., layout=alo),
              FloatText(value=1., layout=alo),
              FloatText(value=0., layout=alo)]
        zs = [FloatText(value=0., layout=alo),
              FloatText(value=0., layout=alo),
              FloatText(value=1., layout=alo)]

        carts = ['x', 'y', 'z']
        for i, ks in enumerate([xs, ys, zs]):
            for j, cart in enumerate(carts):
                for scn in self.scenes:
                    link((scn, 't' + carts[i] + cart), (ks[j], 'value'))

        xbox = HBox(xs, layout=rlo)
        ybox = HBox(ys, layout=rlo)
        zbox = HBox(zs, layout=rlo)

        def _geom(b):
            for idx in self.active_scene_indices:
                self.scenes[idx].geom = not self.scenes[idx].geom

        geom.on_click(_geom)
        mainopts.update([('geom', geom),
                         ('xbox', xbox),
                         ('ybox', ybox),
                         ('zbox', zbox)])

        return mainopts


    def __init__(self, *args, **kwargs):
        super(TensorContainer, self).__init__(*args, uni=False, test=False,
                                              typ=TensorScene, **kwargs)




class TestUniverse(ExatomicBox):
    """A showcase of functional forms used in quantum chemistry."""

    def _update_active(self, b):
        """Control which scenes are controlled by the GUI.
        Additionally align traits with active scenes so that
        the GUI reflects that correct values of active scenes."""

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


    def _field_folder(self, **kwargs):
        """Folder that houses field GUI controls."""

        folder = super(TestUniverse, self)._field_folder(**kwargs)

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
        fopts.observe(_field, names='value')
        folder.insert(1, 'fopts', fopts)
        folder.activate('Hydrogenic', enable=True, update=True)
        folder.move_to_end('alpha', 'iso', 'nx', 'ny', 'nz')

        return folder


    def _init_gui(self, **kwargs):
        """Initialize generic GUI controls and observe callbacks."""

        for scn in self.scenes:
            for attr in ['field_ox', 'field_oy', 'field_oz']:
                setattr(scn, attr, -30.0)
            for attr in ['field_fx', 'field_fy', 'field_fz']:
                setattr(scn, attr, 30.0)
            scn.field = 'Hydrogenic'
            scn.field_iso = 0.0005
            scn.field_kind = '1s'

        mainopts = super(TestUniverse, self)._init_gui()
        mainopts.update([('field', self._field_folder(**kwargs))])
        return mainopts


    def __init__(self, *scenes, **kwargs):
        super(TestUniverse, self).__init__(*scenes, uni=True, test=True,
                                           typ=ExatomicScene, **kwargs)






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
            for idx in self.active_scene_indices:
                self.scenes[idx].frame_idx = c.new
        content['scn_frame'].observe(_scn_frame, names='value')

        jslink((content['playing'], 'value'),
               (content['scn_frame'], 'value'))

        folder = Folder(control, content)
        return folder

    def _field_folder(self, fields, **kwargs):

        folder = super(UniverseWidget, self)._field_folder(**kwargs)
        folder.deactivate('nx', 'ny', 'nz')
        fopts = Dropdown(options=fields)

        def _fopts(c):
            for idx in self.active_scene_indices:
                self.scenes[idx].field_idx = c.new

        fopts.observe(_fopts, names='value')
        folder['fopts'] = fopts
        return folder


    def _iso_folder(self, folder):

        isos = Button(description=' Isosurfaces', icon='cube')
        def _fshow(b):
            for idx in self.active_scene_indices:
                self.scenes[idx].field_show = not self.scenes[idx].field_show
        isos.on_click(_fshow)

        # Move some buttons to the subfolder
        isofolder = Folder(isos, _ListDict([
            ('fopts', folder['fopts']),
            ('alpha', folder.pop('alpha')),
            ('iso', folder.pop('iso'))]))
        isofolder.move_to_end('alpha', 'iso')

        folder.insert(1, 'iso', isofolder, active=True)


    def _contour_folder(self, folder):

        # Make a contour folder
        control = Button(description=' Contours', icon='dot-circle-o')
        def _cshow(b):
            for idx in self.active_scene_indices:
                self.scenes[idx].cont_show = not self.scenes[idx].cont_show
        control.on_click(_cshow)

        content = _ListDict([
            ('fopts', folder['fopts']),
            ('axis', Dropdown(options=['x', 'y', 'z'], value='z')),
            ('num', IntSlider(description='N', min=5, max=20,
                              value=10, step=1)),
            ('lim', IntRangeSlider(description='10**Limits', min=-8,
                                   max=0, step=1, value=[-7, -1])),
            ('val', FloatSlider(description='Value',
                                min=-5, max=5, value=0))])

        def _cont_axis(c):
            for idx in self.active_scene_indices:
                self.scenes[idx].cont_axis = c.new
        def _cont_num(c):
            for idx in self.active_scene_indices:
                self.scenes[idx].cont_num = c.new
        def _cont_lim(c):
            for idx in self.active_scene_indices:
                self.scenes[idx].cont_lim = c.new
        def _cont_val(c):
            for idx in self.active_scene_indices:
                self.scenes[idx].cont_val = c.new

        content['axis'].observe(_cont_axis, names='value')
        content['num'].observe(_cont_num, names='value')
        content['lim'].observe(_cont_lim, names='value')
        content['val'].observe(_cont_val, names='value')
        contour = Folder(control, content)

        folder.insert(2, 'contour', contour, active=True, update=True)



    def _init_gui(self, nframes=1, fields=None, **kwargs):

        mainopts = super(UniverseWidget, self)._init_gui(**kwargs)

        atoms = Button(description=' Fill', icon='adjust', layout=_wlo)
        axis = Button(description=' Axis', icon='arrows-alt', layout=_wlo)

        def _atom_3d(b):
            for idx in self.active_scene_indices:
                self.scenes[idx].atom_3d = not self.scenes[idx].atom_3d
        def _axis(b):
            for idx in self.active_scene_indices:
                self.scenes[idx].axis = not self.scenes[idx].axis

        atoms.on_click(_atom_3d)
        axis.on_click(_axis)
        atoms.active = True
        atoms.disabled = False
        axis.active = True
        atoms.disabled = False

        mainopts.update([('atom_3d', atoms), ('axis', axis),
                         ('frame', self._frame_folder(nframes))])

        if fields is not None:
            folder = self._field_folder(fields, **kwargs)
            self._iso_folder(folder)
            self._contour_folder(folder)
            folder.pop('fopts')

            mainopts.update([('field', folder)])

        return mainopts


    def __init__(self, *unis, **kwargs):

        scenekwargs = kwargs.pop('scenekwargs', {})
        scenekwargs.update({'uni': True, 'test': False})
        atomcolors = scenekwargs.get('atomcolors', None)
        atomradii = scenekwargs.get('atomradii', None)

        fields, masterkwargs = [], []
        for uni in unis:
            unargs, flds = uni_traits(uni,
                                      atomcolors=atomcolors,
                                      atomradii=atomradii)
            fields = flds if len(flds) > len(fields) else fields
            unargs.update(scenekwargs)
            masterkwargs.append(unargs)

        nframes = max((uni.atom.nframes
                      for uni in unis)) if len(unis) else 1

        super(UniverseWidget, self).__init__(*masterkwargs,
                                             nframes=nframes,
                                             fields=fields,
                                             typ=UniverseScene,
                                             **kwargs)
