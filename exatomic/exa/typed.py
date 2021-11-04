# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Strongly Typed Class Attributes
######################################
This module provides a typed attribute class and class decorator/base class for
creating classes with type enforced attributes. Enforcing an attribute's type is
handled by Python's property mechanism (the property's set function checks the
value's type). A simple usage example follows.

.. code-block:: Python

    @typed
    class Foo(object):
        bar = Typed(int, doc="Always an integer type")

The simple code example generates code similar to the following when the module
is executed (i.e. imported).

.. code-block:: python

    class Foo(object):
        @property
        def bar(self):
            return self._bar

        @bar.setter
        def bar(self, value):
            if not isinstance(value, int):
                try:
                    value = int(value)
                except Exception as e:
                    raise TypeError("Cannot convert value") from e
            self._bar = value

The :class:`~exa.typed.Typed` object additionally provides mechanisms for
triggering function calls before and after get, set, and delete, and attempts
automatic conversion (as shown above) for all types supported for a given
attribute.
"""
import pandas as pd
import warnings


def _typed_from_items(items):
    """
    Construct strongly typed attributes (properties) from a dictionary of
    name and :class:`~exa.typed.Typed` object pairs.

    See Also:
        :func:`~exa.typed.typed`
    """
    dct = {}
    for name, attr in items:
        if isinstance(attr, Typed):
            dct[name] = attr(name)
    return dct


def typed(cls):
    """
    Class decorator that updates a class definition with strongly typed
    property attributes.

    See Also:
        If the class will be inherited, use :class:`~exa.typed.TypedClass`.
    """
    for name, attr in _typed_from_items(vars(cls).items()).items():
        setattr(cls, name, attr)
    return cls


def yield_typed(obj_or_cls):
    """
    Generator that yields typed object names of the class (or object's class).

    Args:
        obj_or_cls (object): Class object or instance of class

    Returns:
        name (array): Names of class attributes that are strongly typed
    """
    if not isinstance(obj_or_cls, type):
        obj_or_cls = type(obj_or_cls)
    for attrname in dir(obj_or_cls):
        if hasattr(obj_or_cls, attrname):
            attr = getattr(obj_or_cls, attrname)
            # !!! Important hardcoded value here !!!
            if (isinstance(attr, property) and isinstance(attr.__doc__, str)
                and "__typed__" in attr.__doc__):
                yield attrname


class Typed(object):
    """
    A representation of a strongly typed class attribute.

    .. code-block:: Python

        @typed
        class Strong(object):
            foo = Typed(int, doc="my int")

    The above example creates a class object that has a property-like attribute
    which requires its value to be of type int. Additional arguments provide
    the ability to have the property's getter, setter, and deleter functions call
    other functions or methods of the class. If provided by the class, strongly
    typed attributes created by here automatically attempt to to set themselves
    (see below).

    .. code-block:: Python

        @typed
        class Strong(object):
            _setters = ("_set", )
            foo = Typed(int, doc="my int")

            def _set_foo(self):
                self.foo = 42

    By defining a `_getters` class attribute the strongly typed property knows that,
    if the foo attribute's value (i.e. `_foo`) is not defined (or is defined as None),
    that the property getter should first call the `_set_foo` class method, and after
    it should proceed with getting the property value. Note that `_set_foo` cannot
    accept arguments (it must be 'automatic').

    Args:
        types (iterable, type): Iterable of types or type
        doc (str): Documentation
        autoconv (bool): Attempt automatic type conversion when setting (default true)
        allow_none (bool): As an additional type, allow None (default true)
        pre_set (callable, str): Callable or class method name called before setter
        post_set (callable, str): Callabel or class method name called after setter
        pre_get (callable, str): Callable or class method name called before getter
        pre_del (callable, str): Callable or class method name called before setter
        post_del (callable, str): Callabel or class method name called after setter

    Warning:
        Automatic type conversion (autoconv = true) is not guaranteed to work in
        all cases and is known to fail for non-Python objects such as numpy
        ndarray types: Setting **autoconv** to false is recommened for these cases.

        .. code-block:: python

            Typed(np.ndarray, autoconv=False)    # Do not attempt auto conversion
    """
    def __call__(self, name):
        """
        Construct the property.

        Args:
            name (str): Attribute (property) name

        Returns:
            prop (property): Custom property definition with support for typing
        """
        priv = "_" + name    # Reference to the variable's value

        # The following is a definition of a Python property. Properties have
        # get, set, and delete functions as well as documentation. The variable
        # "this" references the class object instance where the property exists;
        # it does not reference the instance of this ("Typed") class.
        def getter(this):
            # If the variable value (reference by priv) does not exist
            # or is None AND the class has some automatic way of setting the value,
            # set the value first then proceed to getting it.
            if ((not hasattr(this, priv) or getattr(this, priv) is None) and
                hasattr(this, "_setters") and isinstance(this._setters, (list, tuple))):
                for prefix in this._setters:
                    cmd = "{}{}".format(prefix, priv)
                    if hasattr(this, cmd):
                        getattr(this, cmd)()    # Automatic method call
                        if hasattr(this, priv):
                            break
            # Perform pre-get actions (if any)
            if isinstance(self.pre_get, str):
                getattr(this, self.pre_get)()
            elif callable(self.pre_get):
                self.pre_get(this)
            return getattr(this, priv, None)    # Returns None by default

        def setter(this, value):
            # If auto-conversion is on and the value is not the correct type (and
            # also is not None), attempt to convert types
            if self.autoconv and not isinstance(value, self.types) and value is not None:
                for t in self.types:
                    try:
                        value = t(value)
                        break
                    except Exception as e:    # Catch all exceptions but if conversion fails ...
                        if self.verbose:
                            warnings.warn("Conversion of {} (with type {}) failed to type {}\n{}".format(name, type(value), t, str(e)))
                else:          # ... raise a TypeError
                    raise TypeError("Cannot convert object of type {} to any of {}.".format(type(value), self.types))
            # If the value is none and none is not allowed,
            # or the value is some other type (that is not none) and not of a type
            # that is allowed, raise an error.
            elif ((value is None and self.allow_none == False) or
                  (not isinstance(value, self.types) and value is not None)):
                raise TypeError("Object '{}' cannot have type {}, must be of type(s) {}.".format(name, type(value), self.types))
            # Perform pre-set actions (if any)
            if isinstance(self.pre_set, str):
                getattr(this, self.pre_set)()
            elif callable(self.pre_set):
                self.pre_set(this)
            if isinstance(this, (pd.DataFrame, )):
                this[priv] = value
            else:
                setattr(this, priv, value)    # Set the property value
            # Perform post-set actions (if any)
            if isinstance(self.post_set, str):
                getattr(this, self.post_set)()
            elif callable(self.post_set):
                self.post_set(this)

        def deleter(this):
            # Perform pre-del actions (if any)
            if isinstance(self.pre_del, str):
                getattr(this, self.pre_del)()
            elif callable(self.pre_del):
                self.pre_del(this)
            delattr(this, priv)    # Delete the attribute (allows for dynamic naming)
            # Perform post-del actions (if any)
            if isinstance(self.post_del, str):
                getattr(this, self.post_del)()
            elif callable(self.post_del):
                self.post_del(this)

        return property(getter, setter, deleter, doc=self.doc)

    def __init__(self, types, doc=None, autoconv=True, pre_set=None, allow_none=True,
                 post_set=None, pre_get=None, pre_del=None, post_del=None, verbose=False):
        self.types = types if isinstance(types, (tuple, list)) else (types, )
        self.doc = str(doc) + "\n\n__typed__"
        self.autoconv = autoconv
        self.allow_none = allow_none
        self.pre_set = pre_set
        self.post_set = post_set
        self.pre_get = pre_get
        self.pre_del = pre_del
        self.post_del = post_del
        self.verbose = verbose


class TypedMeta(type):
    """
    A metaclass for creating typed attributes which can be used instead of
    the class decorator.

    .. code-block:: Python

        class Foo(metaclass=TypedMeta):
            bar = Typed(int, doc="Always an int")

    See Also:
        :func:`~exa.typed.typed` and :mod:`~exa.core.data`
    """
    def __new__(mcs, name, bases, namespace):
        namespace.update(_typed_from_items(namespace.items()))
        return super(TypedMeta, mcs).__new__(mcs, name, bases, namespace)


class TypedClass(metaclass=TypedMeta):
    """
    A mixin class which can be used to create a class with strongly typed
    attributes.

    .. code-block:: Python

        class Foo(TypedClass):
            bar = Typed(int, doc="Still an int")

    See Also:
        :func:`~exa.typed.typed`
    """
    pass

