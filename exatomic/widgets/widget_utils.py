# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Widget Utilities
#########################
Widget layout and structure.
"""
import six
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


class _ListDict(object):
    """
    Thin wrapper around OrderedDict that allows slicing by position (like list).

    Requires string keys.
    """
    def values(self):
        return self.od.values()

    def keys(self):
        return self.od.keys()

    def items(self):
        return self.od.items()

    def pop(self, key):
        """Pop value."""
        if isinstance(key, six.string_types):
            return self.od.pop(key)
        return self.od.pop(list(self.od.keys())[key])

    def insert(self, idx, key, obj):
        """Insert value at position idx with string key."""
        if not isinstance(key, six.string_types):
            raise TypeError("Key must be type str")
        items = list(self.od.items())
        items.insert(idx, (key, obj))
        self.od = OrderedDict(items)

    def update(self, *args, **kwargs):
        """Update OrderedDict"""
        self.od.update(OrderedDict(*args, **kwargs))

    def __setitem__(self, key, value):
        if not isinstance(key, six.string_types):
            raise TypeError('Must set _ListDict key must be type str.')
        keys = list(self.od.keys())
        if key in keys:
            self.od[key] = value
        else:
            items = list(self.od.items())
            items.append((key, value))
            self.od = OrderedDict(items)

    def __getitem__(self, key):
        if not isinstance(key, (six.string_types, int, slice)):
            raise TypeError('_ListDict slice must be of type str/int/slice.')
        if isinstance(key, six.string_types):
            return self.od[key]
        return list(self.values())[key]

    def __init__(self, *args, **kwargs):
        self.od = OrderedDict(*args, **kwargs)
        if not all((isinstance(key, six.string_types) for key in self.od.keys())):
            raise TypeError('_ListDict keys must be of type str.')

    def __len__(self):
        return len(self.od)

    def __repr__(self):
        return repr(self.od)


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
        #lo = kwargs.pop('layout', None)
        kwargs['layout'] = _glo    # Force global layout
        super(GUIBox, self).__init__(*args, **kwargs)



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
    return _ListDict([('alpha', FloatSlider(description='Opacity', **alims)),
                      ('iso', FloatSlider(**iso_lims)),
                      ('nx', IntSlider(description='Nx', **flims)),
                      ('ny', IntSlider(description='Ny', **flims)),
                      ('nz', IntSlider(description='Nz', **flims))])
