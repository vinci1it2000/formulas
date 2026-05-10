#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2026 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
Python equivalents of database Excel functions.
"""
import functools
import numpy as np
import schedula as sh
from . import (
    __xfilter, wrap_func, Error, XlError, FoundError, flatten, is_number,
    raise_errors
)

FUNCTIONS = {}


def _as_2d(value):
    value = np.asarray(value, object)
    if value.ndim == 0:
        value = value.reshape(1, 1)
    elif value.ndim == 1:
        value = value.reshape(1, -1)
    return value


def _header_key(value):
    return str(value).casefold()


def _field_index(headers, field):
    if field is None or field is sh.EMPTY or field == '':
        return None
    raise_errors(field)
    if is_number(field, xl_return=False):
        index = int(float(field)) - 1
        if 0 <= index < len(headers):
            return index
        raise FoundError(err=Error.errors['#VALUE!'])
    key = _header_key(field)
    for index, header in enumerate(headers):
        if _header_key(header) == key:
            return index
    raise FoundError(err=Error.errors['#VALUE!'])


def _database_parts(database, field, criteria):
    database = _as_2d(database)
    criteria = _as_2d(criteria)
    if database.shape[0] < 2 or criteria.shape[0] < 1:
        raise FoundError(err=Error.errors['#VALUE!'])
    headers = database[0]
    data = database[1:]
    return data, _field_index(headers, field), _criteria_mask(
        headers, data, criteria
    )


def _criteria_mask(headers, data, criteria):
    if not len(data):
        return np.zeros(0, dtype=bool)
    headers = tuple(map(_header_key, headers))
    criteria_headers = criteria[0]
    result = np.zeros(len(data), dtype=bool)
    for criteria_row in criteria[1:]:
        row_mask = np.ones(len(data), dtype=bool)
        for criteria_header, condition in zip(criteria_headers, criteria_row):
            if condition is sh.EMPTY or condition == '':
                continue
            key = _header_key(criteria_header)
            try:
                index = headers.index(key)
            except ValueError:
                raise FoundError(err=Error.errors['#VALUE!'])
            test_range = {'raw': data[:, index]}
            row_mask &= np.asarray(
                __xfilter(test_range, condition), bool
            ).reshape(-1)
        result |= row_mask
    return result


def _selected(database, field, criteria):
    data, index, mask = _database_parts(database, field, criteria)
    if index is None:
        return data[mask]
    return data[mask, index]


def _numeric(values):
    res = []
    for value in flatten(values, None):
        if isinstance(value, XlError):
            raise FoundError(err=value)
        if is_number(value, xl_return=False):
            res.append(float(value))
    return res


def _not_empty(values):
    res = []
    for value in flatten(values, None):
        if isinstance(value, XlError):
            raise FoundError(err=value)
        if value is not sh.EMPTY and value != '':
            res.append(value)
    return res


def xdcount(database, field, criteria, count_all=False):
    values = _selected(database, field, criteria)
    if count_all:
        return len(_not_empty(values))
    return len(_numeric(values))


def xdget(database, field, criteria):
    values = _selected(database, field, criteria)
    if np.asarray(values).ndim > 1:
        raise FoundError(err=Error.errors['#VALUE!'])
    values = list(flatten(values, None))
    if len(values) == 1:
        return values[0]
    raise FoundError(err=Error.errors['#VALUE!' if not values else '#NUM!'])


def xdsum(database, field, criteria):
    return sum(_numeric(_selected(database, field, criteria)))


def xdproduct(database, field, criteria):
    values = _numeric(_selected(database, field, criteria))
    return np.prod(values) if values else 0


def _dstats(database, field, criteria, func, empty_error=Error.errors['#DIV/0!']):
    values = _numeric(_selected(database, field, criteria))
    return func(values) if values else empty_error


FUNCTIONS['DAVERAGE'] = wrap_func(functools.partial(
    _dstats, func=lambda v: np.mean(v)
))
FUNCTIONS['DCOUNT'] = wrap_func(xdcount)
FUNCTIONS['DCOUNTA'] = wrap_func(functools.partial(xdcount, count_all=True))
FUNCTIONS['DGET'] = wrap_func(xdget)
FUNCTIONS['DMAX'] = wrap_func(functools.partial(
    _dstats, func=max, empty_error=0
))
FUNCTIONS['DMIN'] = wrap_func(functools.partial(
    _dstats, func=min, empty_error=0
))
FUNCTIONS['DPRODUCT'] = wrap_func(xdproduct)
FUNCTIONS['DSTDEV'] = wrap_func(functools.partial(
    _dstats,
    func=lambda v: np.std(v, ddof=1) if len(v) > 1 else
    Error.errors['#DIV/0!']
))
FUNCTIONS['DSTDEVP'] = wrap_func(functools.partial(
    _dstats, func=lambda v: np.std(v, ddof=0)
))
FUNCTIONS['DSUM'] = wrap_func(xdsum)
FUNCTIONS['DVAR'] = wrap_func(functools.partial(
    _dstats,
    func=lambda v: np.var(v, ddof=1) if len(v) > 1 else
    Error.errors['#DIV/0!']
))
FUNCTIONS['DVARP'] = wrap_func(functools.partial(
    _dstats, func=lambda v: np.var(v, ddof=0)
))
