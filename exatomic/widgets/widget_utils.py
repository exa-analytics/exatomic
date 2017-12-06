# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Universe Notebook Widget
#########################
"""
from collections import OrderedDict
from ipywidgets import VBox, Layout, FloatSlider, IntSlider

# The GUI box is fixed width
_glo = Layout(flex='0 0 240px')
# The folder is relative to the GUI box
_flo = Layout(width='100%', align_items='flex-end')
# All widgets not in a folder have this layout
_wlo = Layout(width='98%')
# HBoxes within the final scene VBox have this layout
_hboxlo = Layout(flex='1 1 auto', width='auto', height='auto')
# The final VBox containing all scenes has this layout
_vboxlo = Layout(flex='1 1 auto', width='auto', height='auto')
# The box to end all boxes
_bboxlo = Layout(flex='1 1 auto', width='auto', height='auto')
# Box layouts above are separate so it is easier to restyle


class _ListDict(OrderedDict):
    """An OrderedDict that also slices and indexes as a list."""

    def pop(self, key):
        """Pop as a dict or list."""
        try:
            return super(_ListDict, self).pop(key)
        except KeyError:
            key = list(self.keys())[key]
            return super(_ListDict, self).pop(key)

    def insert(self, idx, key, obj):
        """Insert as a list."""
        key = str(key)
        keys = list(self.keys())
        self[key] = obj
        keys.insert(idx, key)
        try:
            # Only python >= 3.2
            for k in keys[idx:]:
                self.move_to_end(k, last=True)
        except AttributeError:
            # Manually reorder
            old = list(self.items())
            for key in self.keys():
                self.pop(key)
            for key, obj in old:
                self[key] = obj


    def __setitem__(self, key, obj):
        if not isinstance(key, str):
            raise TypeError('Must set _ListDict value with key of type str.')
        super(_ListDict, self).__setitem__(key, obj)

    def __getitem__(self, key):
        if isinstance(key, str):
            return super(_ListDict, self).__getitem__(key)
        if isinstance(key, (int, slice)):
            return list(self.values())[key]
        raise TypeError('_ListDict slice must be of type str/int/slice.')

    def __init__(self, *args, **kwargs):
        super(_ListDict, self).__init__(*args, **kwargs)
        if not all((isinstance(key, str) for key in self.keys())):
            raise TypeError('_ListDict keys must be of type str.')


class Folder(VBox):
    """A VBox that shows and hides widgets. For proper
    indentation, instantiate sub-folders before passing to
    super-folders. Should not exist outside of a GUI box."""

    # Cannot also have a keys method -- used by ipywidgets

    def activate(self, *keys, **kwargs):
        """Activate (show) widgets that are not disabled."""
        update = kwargs.pop('update', False)
        enable = kwargs.pop('enable', False)
        keys = self._get(False, True) if not keys else keys
        for key in keys:
            obj = self._controls[key]
            if enable:
                obj.disabled = False
                obj.active = True
            elif not obj.disabled:
                obj.active = True
        if update:
            self._set_gui()


    def deactivate(self, *keys, **kwargs):
        """Deactivate (hide) widgets."""
        active = kwargs.pop('active', False)
        update = kwargs.pop('update', False)
        keys = self._get(True, True) if not keys else keys
        for key in keys:
            if key == 'main': continue
            self._controls[key].active = active
            self._controls[key].disabled = True
        if update:
            self._set_gui()


    def insert(self, idx, key, obj, active=True, update=False):
        """Insert widget into Folder, behaves as list.insert ."""
        obj.layout.width = str(98 - (self.level + 1) * self.indent) + '%'
        self._controls.insert(idx, key, obj)
        if active:
            self.activate(key, enable=True)
        if update:
            self._set_gui()


    def update(self, objs, relayout=False):
        """Update the Folder widgets, behaves as dict.update."""
        if relayout:
            self._relayout(objs)
        self._controls.update(objs)


    def move_to_end(self, *keys):
        """Move widget(s) to the end of the folder."""
        try:
            for key in keys:
                self._controls.move_to_end(key)
        except AttributeError:
            objs = [self._controls.pop(key) for key in keys]
            for key, obj in zip(keys, objs):
                self[key] = obj



    def pop(self, key):
        """Pop a widget from the folder."""
        return self._controls.pop(key)


    def _close(self):
        """Close all widgets in the folder, then the folder."""
        for widget in self._get():
            widget.close()
        self.close()


    def _get(self, active=True, keys=False):
        """Get the widgets in the folder."""
        if keys:
            mit = self._controls.items()
            if active:
                return [key for key, obj in mit if obj.active]
            return [key for key, obj in mit if not obj.active]
        else:
            mit = self._controls.values()
            if active:
                return [obj for obj in mit if obj.active]
            return [obj for obj in mit if not obj.active]


    def _set_gui(self):
        """Update the 'view' of the folder."""
        if self.show:
            self.activate()
            self.children = self._get()
        else:
            self.children = [self._controls['main']]
        self.on_displayed(VBox._fire_children_displayed)


    def _relayout(self, objs):
        """Set layout for widgets in the folder."""
        for obj in objs.values():
            obj.layout = self._slo


    def _init(self, control, content):
        """Set initial layout of primary button and widgets."""

        def _b(b):
            self.show = not self.show
            self._set_gui()
        control.on_click(_b)
        control.active = True
        control.disabled = False
        control.layout = self._plo

        self._controls = _ListDict([('main', control)])

        if content is not None:
            for key, obj in content.items():
                if isinstance(obj, Folder):
                    obj.active = False
                    continue
                obj.layout = self._slo
                if not hasattr(obj, 'active'):
                    obj.active = self.show
                if not hasattr(obj, 'disabled'):
                    obj.disabled = False
            self._controls.update(content)


    def __setitem__(self, key, obj):
        return self._controls.__setitem__(key, obj)


    def __getitem__(self, key):
        return self._controls.__getitem__(key)


    def __init__(self, control, content, **kwargs):
        self.show = kwargs.pop('show', False)
        self.indent = 5
        self.level = kwargs.pop('level', 0)
        pw = 98 - self.level * self.indent
        self._slo = Layout(width=str(pw - self.indent) + '%')
        self._plo = Layout(width=str(pw) + '%')
        self._init(control, content)
        lo = kwargs.pop('layout', None)
        lo = Layout(width='100%', align_items='flex-end')
        super(Folder, self).__init__(
            children=self._get(), layout=lo, **kwargs)
        self.active = True
        self.disabled = False


class GUIBox(VBox):

    def __init__(self, *args, **kwargs):
        lo = kwargs.pop('layout', None)
        super(GUIBox, self).__init__(*args, layout=_glo, **kwargs)



def gui_field_widgets(uni=False, test=False):
    """Return new widgets for field GUI functionality."""

    flims = {'min': 30, 'max': 60,
             'value': 30, 'step': 1,
             'continuous_update': False}
    iso_lims = {'description': 'Iso.',
                'continuous_update': False}
    if uni:
        iso_lims.update({'min': 0.0001, 'max': 0.1,
                         'value': 0.0005, 'step': 0.0005,
                         'readout_format': '.4f'})
    else:
        iso_lims.update({'min': 3.0, 'max': 10.0, 'value': 2.0})
    if uni and not test:
        iso_lims['value'] = 0.03
    alims = {'min': 0.01, 'max': 1.0,
             'value': 1.0, 'step': 0.01}
    return _ListDict(alpha=FloatSlider(description='Opacity', **alims),
                     iso=FloatSlider(**iso_lims),
                     nx=IntSlider(description='Nx', **flims),
                     ny=IntSlider(description='Ny', **flims),
                     nz=IntSlider(description='Nz', **flims))