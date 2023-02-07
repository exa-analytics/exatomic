# -*- coding: utf-8 -*-
# Copyright (c) 2015-2022, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Data Objects
###################################
Data objects are used to store typed data coming from an external source (for
example a file on disk). There are three primary data objects provided by
this module, :class:`~exatomic.exa.core.numerical.Series`, :class:`~exatomic.exa.core.numerical.DataFrame`,
and :class:`~exatomic.exa.core.numerical.Field`. The purpose of these objects is to facilitate
conversion of data into "traits" used in visualization and enforce relationships
between data objects in a given container. Any of the objects provided by this
module may be extended.
"""
import logging
import warnings
import numpy as np
import pandas as pd
from exatomic.exa.core.error import RequiredColumnError


class Numerical(object):
    """
    Base class for :class:`~exatomic.exa.core.numerical.Series`,
    :class:`~exatomic.exa.core.numerical.DataFrame`, and :class:`~exatomic.exa.numerical.Field`
    objects, providing default trait functionality and clean representations
    when present as part of containers.
    """
    @property
    def log(self):
        name = '.'.join([self.__module__, self.__class__.__name__])
        return logging.getLogger(name)

    def slice_naive(self, key):
        """
        Slice a data object based on its index, either by value (.loc) or
        position (.iloc).

        Args:
            key: Single index value, slice, tuple, or list of indices/positionals

        Returns:
            data: Slice of self
        """
        cls = self.__class__
        key = check_key(self, key)
        return cls(self.loc[key])

    def __str__(self):
        return self.__repr__()


class BaseSeries(Numerical):
    """
    Base class for dense and sparse series objects (labeled arrays).

    Attributes:
        _sname (str): May have a required name (default None)
        _iname (str): May have a required index name
        _stype (type): May have a required value type
        _itype (type): May have a required index type
    """
    _metadata = ['name', 'meta']
    # These attributes may be set when subclassing Series
    _sname = None           # Series may have a required name
    _iname = None           # Series may have a required index name
    _stype = None           # Series may have a required value type
    _itype = None           # Series may have a required index type

    def __init__(self, *args, **kwargs):
        meta = kwargs.pop('meta', None)
        super(BaseSeries, self).__init__(*args, **kwargs)
        if hasattr(self, "name") and hasattr(self, "_sname") and hasattr(self, "_iname"):
            if self._sname is not None and self.name != self._sname:
                if self.name is not None:
                    warnings.warn("Object's name changed")
                self.name = self._sname
            if self._iname is not None and self.index.name != self._iname:
                if self.index.name is not None:
                    warnings.warn("Object's index name changed")
                self.index.name = self._iname
        self.meta = meta


class BaseDataFrame(Numerical):
    """
    Base class for dense and sparse dataframe objects (labeled matrices).

    Note:
        If the _cardinal attribute is populated, it will automatically be added
        to the _categories and _columns attributes.

    Attributes:
        _cardinal (tuple): Tuple of column name and raw type that acts as foreign key to index of another table
        _index (str): Name of index (may be used as foreign key in another table)
        _columns (list): Required columns
        _categories (dict): Dict of column names, raw types that if present will be converted to and from categoricals automatically
    """
    _metadata = ['name', 'meta']
    _cardinal = None     # Tuple of column name and raw type that acts as foreign key to index of another table
    _index = None      # Name of index (may be used as foreign key in another table)
    _columns = []      # Required columns
    _categories = {}   # Dict of column names, raw types that if present will be converted to and from categoricals automatically

    def cardinal_groupby(self):
        """
        Group this object on it cardinal dimension (_cardinal).

        Returns:
            grpby: Pandas groupby object (grouped on _cardinal)
        """
        g, t = self._cardinal
        self[g] = self[g].astype(t)
        grpby = self.groupby(g)
        self[g] = self[g].astype('category')
        return grpby

    def slice_cardinal(self, key):
        """
        Get the slice of this object by the value or values of the cardinal
        dimension.
        """
        cls = self.__class__
        key = check_key(self, key, cardinal=True)
        return cls(self[self[self._cardinal[0]].isin(key)])

    def __init__(self, *args, **kwargs):
        meta = kwargs.pop('meta', None)
        super(BaseDataFrame, self).__init__(*args, **kwargs)
        self.meta = meta


class Series(BaseSeries, pd.Series):
    """
    A labeled array.

    .. code-block:: Python

        class MySeries(exatomic.exa.core.numerical.Series):
            _sname = 'data'        # series default name
            _iname = 'data_index'  # series default index name

        seri = MySeries(np.random.rand(10**5))
    """
    @property
    def _constructor(self):
        return Series

    def copy(self, *args, **kwargs):
        """
        Make a copy of this object.

        See Also:
            For arguments and description of behavior see `pandas docs`_.

        .. _pandas docs: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.copy.html
        """
        cls = self.__class__    # Note that type conversion does not perform copy
        return cls(pd.Series(self).copy(*args, **kwargs))


class DataFrame(BaseDataFrame, pd.DataFrame):
    """
    A data table

    .. code-block:: Python

        class MyDF(exatomic.exa.core.numerical.DataFrame):
            _cardinal = ('cardinal', int)
            _index = 'mydf_index'
            _columns = ['x', 'y', 'z', 'symbol']
            _categories = {'symbol': str}
    """
    _constructor_sliced = Series

    @property
    def _constructor(self):
        return DataFrame

    def copy(self, *args, **kwargs):
        """
        Make a copy of this object.

        See Also:
            For arguments and description of behavior see `pandas docs`_.

        .. _pandas docs: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.copy.html
        """
        cls = self.__class__    # Note that type conversion does not perform copy
        return cls(pd.DataFrame(self).copy(*args, **kwargs))

    def _revert_categories(self, inplace=True):
        """
        Inplace conversion to categories.
        """
        if inplace:
            for column, dtype in self._categories.items():
                if column in self.columns:
                    self[column] = self[column].astype(dtype)
        else:
            copy = self.copy()
            for column, dtype in copy._categories.items():
                if column in copy.columns:
                    copy[column] = copy[column].astype(dtype)
            return copy

    def _set_categories(self):
        """
        Inplace conversion from categories.
        """
        for column, _ in self._categories.items():
            if column in self.columns:
                self[column] = self[column].astype('category')

    def __init__(self, *args, **kwargs):
        super(DataFrame, self).__init__(*args, **kwargs)
        self.log.debug('shape: {}'.format(self.shape))
        if self._cardinal is not None:
            self._categories[self._cardinal[0]] = self._cardinal[1]
            self._columns.append(self._cardinal[0])
        self._set_categories()
        if len(self) > 0:
            name = self.__class__.__name__
            if self._columns:
                missing = set(self._columns).difference(self.columns)
                if missing:
                    raise RequiredColumnError(missing, name)
            if self.index.name != self._index and self._index is not None:
                if self.index.name is not None and self.index.name.decode('utf-8') != self._index:
                    warnings.warn("Object's index name changed from {} to {}".format(self.index.name, self._index))
                self.index.name = self._index


class Field(DataFrame):
    """
    A field is defined by field data and field values. Field data defines the
    discretization of the field (i.e. its origin in a given space, number of
    steps/step spaceing, and endpoint for example). Field values can be scalar
    (series) and/or vector (dataframe) data defining the magnitude and/or direction
    at each given point.

    Note:
        The convention for generating the discrete field data and ordering of
        the field values must be the same (e.g. discrete field points are
        generated x, y, then z and scalar field values are a series object
        ordered looping first over x then y, then z).

    In addition to the :class:`~exatomic.exa.core.numerical.DataFrame` attributes, this object
    has the following:
    """
    @property
    def _constructor(self):
        return Field

    def copy(self, *args, **kwargs):
        """
        Make a copy of this object.

        Note:
            Copies both field data and field values.

        See Also:
            For arguments and description of behavior see `pandas docs`_.

        .. _pandas docs: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.copy.html
        """
        cls = self.__class__    # Note that type conversion does not perform copy
        data = pd.DataFrame(self).copy(*args, **kwargs)
        values = [field.copy() for field in self.field_values]
        return cls(data, field_values=values)

    def memory_usage(self):
        """
        Get the combined memory usage of the field data and field values.
        """
        data = super(Field, self).memory_usage()
        values = 0
        for value in self.field_values:
            values += value.memory_usage()
        data['field_values'] = values
        return data

    def slice_naive(self, key):
        """
        Naively (on index) slice the field data and values.

        Args:
            key: Int, slice, or iterable to select data and values

        Returns:
            field: Sliced field object
        """
        cls = self.__class__
        key = check_key(self, key)
        enum = pd.Series(range(len(self)))
        enum.index = self.index
        values = self.field_values[enum[key].values]
        data = self.loc[key]
        return cls(data, field_values=values)

    #def slice_cardinal(self, key):
    #    cls = self.__class__
    #    grpby = self.cardinal_groupby()

    def __init__(self, *args, **kwargs):
        # The following check allows creation of a single field (whose field data
        # comes from a series object and field values from another series object).
        field_values = kwargs.pop("field_values", None)
        if args and isinstance(args[0], pd.Series):
            args = (args[0].to_frame().T, )
        super(Field, self).__init__(*args, **kwargs)
        self._metadata = ['field_values']
        if isinstance(field_values, (list, tuple, np.ndarray)):
            self.field_values = [Series(v) for v in field_values]    # Convert type for nice repr
        elif field_values is None:
            self.field_values = []
        elif isinstance(field_values, pd.Series):
            self.field_values = [Series(field_values)]
        else:
            raise TypeError("Wrong type for field_values with type {}".format(type(field_values)))
        for i in range(len(self.field_values)):
            self.field_values[i].name = i
        self.log.info('contains {} fields'.format(len(self.field_values)))


class Field3D(Field):
    """
    Dataframe for storing dimensions of a scalar or vector field of 3D space.

    +-------------------+----------+-------------------------------------------+
    | Column            | Type     | Description                               |
    +===================+==========+===========================================+
    | nx                | int      | number of grid points in x                |
    +-------------------+----------+-------------------------------------------+
    | ny                | int      | number of grid points in y                |
    +-------------------+----------+-------------------------------------------+
    | nz                | int      | number of grid points in z                |
    +-------------------+----------+-------------------------------------------+
    | ox                | float    | field origin point in x                   |
    +-------------------+----------+-------------------------------------------+
    | oy                | float    | field origin point in y                   |
    +-------------------+----------+-------------------------------------------+
    | oz                | float    | field origin point in z                   |
    +-------------------+----------+-------------------------------------------+
    | xi                | float    | First component in x                      |
    +-------------------+----------+-------------------------------------------+
    | xj                | float    | Second component in x                     |
    +-------------------+----------+-------------------------------------------+
    | xk                | float    | Third component in x                      |
    +-------------------+----------+-------------------------------------------+
    | yi                | float    | First component in y                      |
    +-------------------+----------+-------------------------------------------+
    | yj                | float    | Second component in y                     |
    +-------------------+----------+-------------------------------------------+
    | yk                | float    | Third component in y                      |
    +-------------------+----------+-------------------------------------------+
    | zi                | float    | First component in z                      |
    +-------------------+----------+-------------------------------------------+
    | zj                | float    | Second component in z                     |
    +-------------------+----------+-------------------------------------------+
    | zk                | float    | Third component in z                      |
    +-------------------+----------+-------------------------------------------+

    Note:
        Each field should be flattened into an N x 1 (scalar) or N x 3 (vector)
        series or dataframe respectively. The orientation of the flattening
        should have x as the outer loop and z values as the inner loop (for both
        cases). This is sometimes called C-major or C-style order, and has
        the last index changing the fastest and the first index changing the
        slowest.

    See Also:
        :class:`~exatomic.exa.core.numerical.Field`
    """
    _columns = ['nx', 'ny', 'nz', 'ox', 'oy', 'oz', 'xi', 'xj', 'xk',
                'yi', 'yj', 'yk', 'zi', 'zj', 'zk']

    @property
    def _constructor(self):
        return Field3D


def check_key(data_object, key, cardinal=False):
    """
    Update the value of an index key by matching values or getting positionals.
    """
    itype = (int, np.int32, np.int64)
    if not isinstance(key, itype + (slice, tuple, list, np.ndarray)):
        raise KeyError("Unknown key type {} for key {}".format(type(key), key))
    keys = data_object.index.values
    if cardinal and data_object._cardinal is not None:
        keys = data_object[data_object._cardinal[0]].unique()
    elif isinstance(key, itype) and (key in keys or key < 0):
        key = keys[key]
        if isinstance(key, itype):
            key = [key]
        else:
            key = list(sorted(key))
    elif isinstance(key, itype):
        key = [key]
    elif isinstance(key, slice):
        key = list(sorted(keys[key]))
    elif isinstance(key, (tuple, list, pd.Index)) and not np.all(k in keys for k in key):
        key = list(sorted(keys[key]))
    return key


class SparseDataFrame(BaseDataFrame):
    @property
    def _constructor(self):
        return SparseDataFrame
