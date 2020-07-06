# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Universe Notebook Widget
#########################
To visualize a universe containing atoms, molecules, orbitals, etc., do
the following in a Jupyter notebook environment.

.. code-block:: Python

    exatomic.UniverseWidget(u)    # type(u) is exatomic.core.universe.Universe

"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
#from traitlets import Unicode, link
from traitlets import Unicode
from ipywidgets import (Button, Dropdown, jslink, register, VBox, HBox,
                        IntSlider, IntRangeSlider, FloatSlider, Play,
                        FloatText, Layout, Text, Label, Select, Output)
from .widget_base import (ExatomicScene, UniverseScene,
                          TensorScene, ExatomicBox)
from .widget_utils import _wlo, _ListDict, Folder
from .traits import uni_traits
from exatomic.core.tensor import Tensor
from IPython.display import display_html
from exa.util.units import Length
import pandas as pd
from numpy import sqrt


class DemoContainer(ExatomicBox):
    """A proof-of-concept mixing GUI controls with a three.js scene."""
    def _field_folder(self, **kwargs):
        """Folder that houses field GUI controls."""
        folder = super(DemoContainer, self)._field_folder(**kwargs)
        fopts = Dropdown(options=['null', 'Sphere', 'Torus', 'Ellipsoid'])
        fopts.active = True
        fopts.disabled = False
        def _field(c):
            for scn in self.active():
                scn.field = c.new
        fopts.observe(_field, names='value')
        folder.insert(1, 'options', fopts)
        return folder

    def _init_gui(self, **kwargs):
        """Initialize generic GUI controls and observe callbacks."""
        mainopts = super(DemoContainer, self)._init_gui()
        geom = Button(icon='gear', description=' Mesh', layout=_wlo)
        def _geom(b):
            for scn in self.active():
                scn.geom = not scn.geom
        geom.on_click(_geom)
        mainopts.update([('geom', geom),
                         ('field', self._field_folder(**kwargs))])
        return mainopts

    def __init__(self, *scenes, **kwargs):
        super(DemoContainer, self).__init__(*scenes,
                                            uni=False,
                                            #test=True,
                                            typ=ExatomicScene,
                                            **kwargs)

@register
class TensorContainer(ExatomicBox):
    """
    A simple container to implement cartesian tensor visualization.

    Args:
        file_path (string): Takes a file path name to pass through the Tensor.from_file function. Default to None.
    """
    _model_name = Unicode('TensorContainerModel').tag(sync=True)
    _view_name = Unicode('TensorContainerView').tag(sync=True)

    def _update_active(self, b):
        """
        Control which scenes are controlled by the GUI.

        Additionally align traits with active scenes so that
        the GUI reflects that correct values of active scenes.
        """
        super(TensorContainer, self)._update_active(b)
        scns = self.active()
        if not scns or len(scns) == 1: return
        carts = ['x', 'y', 'z']
        cache = {}
        for i in carts:
            for j in carts:
                tij = 't' + i + j
                cache[tij] = getattr(scns[0], tij)
        for tij, val in cache.items():
            for scn in scns[1:]:
                setattr(scn, tij, val)

    def _init_gui(self, **kwargs):
        """Initialize generic GUI controls and observe callbacks."""
        mainopts = super(TensorContainer, self)._init_gui(**kwargs)
        scn = self.scenes[0]
        alo = Layout(width='74px')
        rlo = Layout(width='235px')
        if self._df is not None:
            scn.txx = self._df.loc[0,'xx']
            scn.txy = self._df.loc[0,'xy']
            scn.txz = self._df.loc[0,'xz']
            scn.tyx = self._df.loc[0,'yx']
            scn.tyy = self._df.loc[0,'yy']
            scn.tyz = self._df.loc[0,'yz']
            scn.tzx = self._df.loc[0,'zx']
            scn.tzy = self._df.loc[0,'zy']
            scn.tzz = self._df.loc[0,'zz']
        xs = [FloatText(value=scn.txx , layout=alo),
              FloatText(value=scn.txy , layout=alo),
              FloatText(value=scn.txz , layout=alo)]
        ys = [FloatText(value=scn.tyx , layout=alo),
              FloatText(value=scn.tyy , layout=alo),
              FloatText(value=scn.tyz , layout=alo)]
        zs = [FloatText(value=scn.tzx , layout=alo),
              FloatText(value=scn.tzy , layout=alo),
              FloatText(value=scn.tzz , layout=alo)]
        opt = [0] if self._df is None else [int(x) for x in self._df.index.values]
        tensorIndex = Dropdown(options=opt, value=opt[0], layout=rlo)
        tdxlabel = Label(value='Select the tensor index:')
        def _x0(c):
            for scn in self.active(): scn.txx = c.new
        def _x1(c):
            for scn in self.active(): scn.txy = c.new
        def _x2(c):
            for scn in self.active(): scn.txz = c.new
        def _y0(c):
            for scn in self.active(): scn.tyx = c.new
        def _y1(c):
            for scn in self.active(): scn.tyy = c.new
        def _y2(c):
            for scn in self.active(): scn.tyz = c.new
        def _z0(c):
            for scn in self.active(): scn.tzx = c.new
        def _z1(c):
            for scn in self.active(): scn.tzy = c.new
        def _z2(c):
            for scn in self.active(): scn.tzz = c.new
        xs[0].observe(_x0, names='value')
        xs[1].observe(_x1, names='value')
        xs[2].observe(_x2, names='value')
        ys[0].observe(_y0, names='value')
        ys[1].observe(_y1, names='value')
        ys[2].observe(_y2, names='value')
        zs[0].observe(_z0, names='value')
        zs[1].observe(_z1, names='value')
        zs[2].observe(_z2, names='value')
        rlo = Layout(width='234px')
        xbox = HBox(xs, layout=rlo)
        ybox = HBox(ys, layout=rlo)
        zbox = HBox(zs, layout=rlo)
        geom = Button(icon='cubes', description=' Geometry', layout=_wlo)

        def _change_tensor(tdx=0):
            carts = ['x','y','z']
            for i, bra in enumerate(carts):
                for j, ket in enumerate(carts):
                    if i == 0:
                        xs[j].value = self._df.loc[tdx,bra+ket]
                    elif i == 1:
                        ys[j].value = self._df.loc[tdx,bra+ket]
                    elif i == 2:
                        zs[j].value = self._df.loc[tdx,bra+ket]

        def _geom(b):
            for scn in self.active(): scn.geom = not scn.geom

        def _tdx(c):
            for scn in self.active(): scn.tdx = c.new
            _change_tensor(c.new)

        geom.on_click(_geom)
        tensorIndex.observe(_tdx, names="value")
        mainopts.update([('geom', geom),
                         ('tlbl', tdxlabel),
                         ('tidx', tensorIndex),
                         ('xbox', xbox),
                         ('ybox', ybox),
                         ('zbox', zbox)])
        return mainopts

    def __init__(self, *args, **kwargs):
        file_path = kwargs.pop("file_path", None)
        if file_path is not None:
            self._df = Tensor.from_file(file_path)
        else:
            self._df = None
        super(TensorContainer, self).__init__(*args,
                                              uni=False,
                                              #test=False,
                                              typ=TensorScene,
                                              **kwargs)


class DemoUniverse(ExatomicBox):
    """A showcase of functional forms used in quantum chemistry."""
    def _update_active(self, b):
        """
        Control which scenes are controlled by the GUI.

        Additionally align traits with active scenes so that
        the GUI reflects that correct values of active scenes.
        """
        super(DemoUniverse, self)._update_active(b)
        scns = self.active()
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
        folder = super(DemoUniverse, self)._field_folder(**kwargs)
        uni_field_lists = _ListDict([
            ('Hydrogenic', ['1s',   '2s',   '2px', '2py', '2pz',
                            '3s',   '3px',  '3py', '3pz',
                            '3d-2', '3d-1', '3d0', '3d+1', '3d+2']),
            ('Gaussian', ['s', 'px', 'py', 'pz', 'd200', 'd110',
                          'd101', 'd020', 'd011', 'd002', 'f300',
                          'f210', 'f201', 'f120', 'f111', 'f102',
                          'f030', 'f021', 'f012', 'f003']),
            ('SolidHarmonic', [str(i) for i in range(8)])])
        kind_widgets = _ListDict([
            (key, Dropdown(options=vals))
            for key, vals in uni_field_lists.items()])
        ml_widgets = _ListDict([
            (str(l), Dropdown(options=[str(i) for i in range(-l, l+1)]))
            for l in range(8)])
        fopts = list(uni_field_lists.keys())
        folder.update(kind_widgets, relayout=True)
        folder.update(ml_widgets, relayout=True)

        def _field(c):
            fk = uni_field_lists[c.new][0]
            for scn in self.active():
                scn.field = c.new
                scn.field_kind = fk
            folder.deactivate(c.old)
            folder[c.new].value = fk
            folder.activate(c.new, enable=True)
            if c.new == 'SolidHarmonic':
                folder.activate(fk, enable=True)
            else:
                aml = [key for key in folder._get(keys=True)
                       if key.isnumeric()]
                if aml:
                    folder.deactivate(*aml)
            folder._set_gui()

        def _field_kind(c):
            for scn in self.active():
                scn.field_kind = c.new
                if scn.field == 'SolidHarmonic':
                    scn.field_ml = folder[c.new].options[0]
                    folder.activate(c.new, enable=True)
                    folder.deactivate(c.old)
                    if scn.field_ml != '0':
                        folder.deactivate('0')
                else:
                    aml = [i for i in folder._get(keys=True)
                           if i.isnumeric()]
                    if aml:
                        folder.deactivate(*aml)
            folder._set_gui()

        def _field_ml(c):
            for scn in self.active(): scn.field_ml = c.new

        for key, obj in kind_widgets.items():
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
        mainopts = super(DemoUniverse, self)._init_gui()
        mainopts.update([('field', self._field_folder(**kwargs))])
        return mainopts


    def __init__(self, *scenes, **kwargs):
        super(DemoUniverse, self).__init__(*scenes, uni=True, #test=True,
                                           typ=ExatomicScene, **kwargs)


@register
class UniverseWidget(ExatomicBox):
    """
    Visualize a :class:`~exatomic.core.universe.Universe`.

    .. code-block:: python

        u = exatomic.Universe.load(exatomic.base.resource("adf-lu-valid.hdf5"))
        scenekwargs = dict(atomcolors=dict(Lu="black"))
        exatomic.UniverseWidget(u, scenekwargs=scenekwargs)    # In Jupyter notebook

        scenekwargs = dict(atomcolors=dict(Lu="#f442f1"), atomradii=dict(Lu=1.0))
        exatomic.UniverseWidget(u, scenekwargs=scenekwargs)    # In Jupyter notebook

    Args:
        uni: The Universe object
        scenekwargs (dict): Keyword args to be passed to :class:`~exatomic.widgets.widget_base.ExatomicScene`
    """
    def _frame_folder(self, nframes):
        playable = bool(nframes <= 1)
        flims = dict(min=0, max=nframes-1, step=1, value=0)
        control = Button(description=' Animate', icon='play')
        content = _ListDict([
            ('playing', Play(disabled=playable, **flims)),
            ('scn_frame', IntSlider(description='Frame', **flims))])

        def _scn_frame(c):
            for scn in self.active(): scn.frame_idx = c.new

        content['scn_frame'].observe(_scn_frame, names='value')
        content['playing'].active = False
        jslink((content['playing'], 'value'),
               (content['scn_frame'], 'value'))
        folder = Folder(control, content)
        return folder


    def _field_folder(self, fields, **kwargs):
        folder = super(UniverseWidget, self)._field_folder(**kwargs)
        folder.deactivate('nx', 'ny', 'nz')
        fopts = Dropdown(options=fields)
        def _fopts(c):
            for scn in self.active(): scn.field_idx = c.new
        fopts.observe(_fopts, names='value')
        folder['fopts'] = fopts
        return folder

    def _iso_folder(self, folder):
        isos = Button(description=' Isosurfaces', icon='cube')
        def _fshow(b):
            for scn in self.active(): scn.field_show = not scn.field_show
        isos.on_click(_fshow)
        isofolder = Folder(isos, _ListDict([
            ('fopts', folder['fopts']),
            ('alpha', folder.pop('alpha')),
            ('iso', folder.pop('iso'))]))
        isofolder.move_to_end('alpha', 'iso')
        folder.insert(1, 'iso', isofolder, active=True)


    def _contour_folder(self, folder):
        control = Button(description=' Contours', icon='dot-circle-o')
        def _cshow(b):
            for scn in self.active(): scn.cont_show = not scn.cont_show
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
            for scn in self.active(): scn.cont_axis = c.new
        def _cont_num(c):
            for scn in self.active(): scn.cont_num = c.new
        def _cont_lim(c):
            for scn in self.active(): scn.cont_lim = c.new
        def _cont_val(c):
            for scn in self.active(): scn.cont_val = c.new
        content['axis'].observe(_cont_axis, names='value')
        content['num'].observe(_cont_num, names='value')
        content['lim'].observe(_cont_lim, names='value')
        content['val'].observe(_cont_val, names='value')
        contour = Folder(control, content)
        folder.insert(2, 'contour', contour, active=True, update=True)

    def _tensor_folder(self):
        alo = Layout(width='70px')
        rlo = Layout(width='220px')
        scale =  FloatSlider(max=10.0, step=0.001, readout=True, value=1.0)
        tens = Button(description=' Tensor', icon='bank')
        def _tens(c):
            for scn in self.active():
                scn.tens = not scn.tens
        def _scale(c):
            for scn in self.active(): scn.scale = c.new
        tens.on_click(_tens)
        scale.observe(_scale, names='value')
        content = _ListDict([
                ('scale', scale),
                ])
        return Folder(tens, content)

    def _fill_folder(self):
        atoms = Button(description=' Fill', icon='adjust', layout=_wlo)
        opt = ['Ball and Stick',
               'Van Der Waals Spheres',
               'Covalent Spheres']
               #'High Performance']#,'Stick']
        fill = Select(options=opt, value=opt[0], layout=_wlo)
        bond_r = FloatSlider(max=1.0,description='Bond Radius')

        def _atoms(c):
            for scn in self.active(): scn.atom_3d = not scn.atom_3d

        def _fill(c):
            for scn in self.active(): scn.fill_idx = c.new
            bond_r.disabled = True if c.new == 1 else False

        def _bond_r(c):
            for scn in self.active(): scn.bond_r = c.new

        atoms.on_click(_atoms)
        fill.observe(_fill, names='index')
        bond_r.observe(_bond_r, names='value')
        content = _ListDict([
                    ('opt', fill),
                    ('bond_r', bond_r)
                    ])
        return Folder(atoms, content)

    def _update_output(self, out):
        out.clear_output()
        idx = {}
        df = pd.DataFrame([])
        for sdx, scn in enumerate(self.active()):
            if not scn.selected:
                continue
            elif len(scn.selected['idx']) == 0:
                continue
            idx[sdx] = [int(''.join(filter(lambda x: x.isdigit(), i))) for i in scn.selected['idx']]
            if len(idx[sdx])%2 != 0:
                raise ValueError("Must select an even number of atoms. Last selected atom has been truncated.")
            atom_coords = self._uniatom[sdx].groupby('frame').get_group(scn.frame_idx). \
                               reset_index(drop=True).loc[[i for i in idx[sdx]], ['x', 'y', 'z']]
            atom_coords.set_index([[i for i in range(len(atom_coords))]], inplace=True)
            distance = [self._get_distance(atom_coords.loc[i, ['x', 'y', 'z']].values,
                        atom_coords.loc[i+1, ['x', 'y', 'z']].values)
                        for i in range(0, len(atom_coords), 2)]
            distance = [i*Length['au', 'Angstrom'] for i in distance]
            with out:
                df = pd.concat([df,pd.DataFrame([[distance[int(i/2)], idx[sdx][i], idx[sdx][i+1], sdx]
                                    for i in range(0, len(idx[sdx]), 2)],
                                    columns=["dr (Angs.)", "adx0", "adx1", "scene"])])
        with out:
            display_html(df.to_html(), raw=True)

    @staticmethod
    def _get_distance(x, y):
        return sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2 +(x[2]-y[2])**2)

    def _distanceBox(self):
        #TODO: Find way to automatically update the table when there
        #      is a change in the selected atoms on the javascript side.
        atom_df = Button(description='Distance', layout=_wlo)
        clear_selected = Button(description='Clear Sel.')
        get_selected = Button(description='Update out')
        select_opt = HBox([clear_selected, get_selected], layout=_wlo)
        out = Output(layout=_wlo)

        #selected = Dict().tag(sync=True)

        def _atom_df(c):
            c.value = not c.value
            if c.value:
                self._update_output(out)
                #link((self.scenes[0].selected, 'value'), (selected, 'value'))
            else:
                out.clear_output()

        def _clear_selected(c):
            for scn in self.active(): scn.clear_selected = not scn.clear_selected
            out.clear_output()

        def _get_selected(c):
            self._update_output(out)

        #selected = List(Dict()).tag(sync=True)
        #for scn in self.active():
        #    selected.append(scn.selected)
        #
        #@observe('selected')
        #def _selected(c):
        #    print(c)

        atom_df.on_click(_atom_df)
        atom_df.value = False
        clear_selected.on_click(_clear_selected)
        get_selected.on_click(_get_selected)
        content = _ListDict([('out', out),
                             ('select_opt', select_opt)
                            ])
        return Folder(atom_df, content)

    def _init_gui(self, **kwargs):
        nframes = kwargs.pop("nframes", 1)
        fields = kwargs.pop("fields", None)
        tensors = kwargs.pop("tensors", None)
#        freq = kwargs.pop("freq", None)
        mainopts = super(UniverseWidget, self)._init_gui(**kwargs)
        atoms = Button(description=' Fill', icon='adjust', layout=_wlo)
        axis = Button(description=' Axis', icon='arrows-alt', layout=_wlo)

        def _atom_3d(b):
            for scn in self.active(): scn.atom_3d = not scn.atom_3d

        def _axis(b):
            for scn in self.active(): scn.axis = not scn.axis

        atoms.on_click(_atom_3d)
        axis.on_click(_axis)
        atoms.active = True
        atoms.disabled = False
        axis.active = True
        axis.disabled = False
        mainopts.update([('atom_3d', self._fill_folder()),
                         ('axis', axis),
                         ('frame', self._frame_folder(nframes))])
        if fields is not None:
            folder = self._field_folder(fields, **kwargs)
            self._iso_folder(folder)
            self._contour_folder(folder)
            folder.pop('fopts')
            mainopts.update([('field', folder)])

        if tensors is not None:
            mainopts.update([('tensor', self._tensor_folder())])

        #print(freq)
        #if freq is not None:
        #    print("Inside frequency")

        mainopts.update([('distance', self._distanceBox())])

        return mainopts

    def __init__(self, *unis, **kwargs):
        scenekwargs = kwargs.pop('scenekwargs', {})
        #scenekwargs.update({'uni': True, 'test': False})
        scenekwargs.update({'uni': True})
        atomcolors = scenekwargs.get('atomcolors', None)
        atomradii = scenekwargs.get('atomradii', None)
        atomlabels = scenekwargs.get('atomlabels', None)
#        fields, masterkwargs, tens, freq = [], [], [], []
        fields, masterkwargs, tens = [], [], []
        self._uniatom = []
        for uni in unis:
            self._uniatom.append(uni.atom)
            unargs, flds, tens = uni_traits(uni,
                                                  atomcolors=atomcolors,
                                                  atomradii=atomradii,
                                                  atomlabels=atomlabels)
            #unargs, flds, tens, freq = uni_traits(uni,
            #                                      atomcolors=atomcolors,
            #                                      atomradii=atomradii,
            #                                      atomlabels=atomlabels)
            #tensors = tens
            fields = flds if len(flds) > len(fields) else fields
            unargs.update(scenekwargs)
            masterkwargs.append(unargs)
        nframes = max((uni.atom.nframes
                      for uni in unis)) if len(unis) else 1
        super(UniverseWidget, self).__init__(*masterkwargs,
                                             uni=True,
                                             #test=False,
                                             nframes=nframes,
                                             fields=fields,
                                             typ=UniverseScene,
                                             tensors=tens,
                                             #freq=freq,
                                             **kwargs)
