#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
It provides Ranges class.
"""
import itertools
import numpy as np
from .tokens.operand import _re_range, range2parts, _index2col, maxsize, Error
from .errors import RangeValueError
from .formulas import Array
import schedula as sh
import functools


def _has_same_sheet(x, y):
    try:
        return x['excel'] == y['excel'] and x['sheet'] == y['sheet']
    except KeyError:
        return False


def _have_intersect(x, y):
    if _has_same_sheet(x, y):
        n1, n2 = max(y['n1'], x['n1']), min(y['n2'], x['n2'])
        if n1 <= n2:
            r1 = max(int(y['r1']), int(x['r1']))
            r2 = min(int(y['r2']), int(x['r2']))
            if r1 <= r2:
                return {
                    'excel': x['excel'], 'sheet': x['sheet'], 'n1': n1,
                    'r1': str(r1), 'n2': n2, 'r2': str(r2)
                }
    return {}


def _single_intersect(format_range, x, y):
    z = _have_intersect(x, y)
    if z:
        return dict(format_range(('name', 'n1', 'n2'), **z))
    return {}


def _split(base, rng, intersect=None, format_range=range2parts):
    z = _have_intersect(base, rng)
    if not z:
        return rng,

    if intersect is not None:
        intersect.update(z)

    ranges = []
    rng = sh.selector(('excel', 'sheet', 'n1', 'n2', 'r1', 'r2'), rng)
    it = ('n1', 'n2', 1), ('n2', 'n1', -1), ('r1', 'r2', 1), ('r2', 'r1', -1)
    for i, j, n in it:
        if z[i] != rng[i]:
            if j[0] == 'r':
                r = sh.combine_dicts(rng, {j: str(int(z[i]) - n)})
            else:
                r = sh.combine_dicts(rng, {j: z[i] - n})
            r = dict(format_range(('name', 'n1', 'n2'), **r))
            ranges.append(r)
            rng[i] = z[i]

    return tuple(ranges)


def _intersect(rng, ranges, format_range=range2parts):
    it = map(functools.partial(_single_intersect, format_range, rng), ranges)
    return tuple(r for r in it if r)


def _merge_raw_update(base, rng):
    if _has_same_sheet(base, rng):
        if base['n1'] == rng['n2'] and int(base['r2']) + 1 >= int(rng['r1']):
            base['r2'] = rng['r2']
            return True


def _merge_col_update(base, rng):
    if _has_same_sheet(base, rng):
        if (base['n2'] + 1) == rng['n1']:
            if base['r1'] == rng['r1'] and base['r2'] == rng['r2']:
                base['n2'] = rng['n2']
                return True


def _get_indices_intersection(base, i):
    r, c = int(base['r1']), base['n1']
    r = slice(int(i['r1']) - r, int(i['r2']) - r + 1)
    c = slice(i['n1'] - c, i['n2'] - c + 1)
    return r, c


def _assemble_values(base, values, empty=''):
    res = np.empty(_shape(**base), object)
    res[:, :] = empty
    for k, (rng, value) in sorted(values.items()):
        ist = _have_intersect(base, rng)
        if ist:
            br, bc = _get_indices_intersection(base, ist)
            rr, rc = _get_indices_intersection(rng, ist)
            res[br, bc] = value[rr, rc]
    return res


# noinspection PyUnusedLocal
def _shape(n1, n2, r1, r2, **kw):
    r1, r2 = int(r1), int(r2)
    r = -1 if r1 == 0 or r2 == maxsize else (r2 - r1 + 1)
    c = -1 if n1 == 0 or n2 == maxsize else (n2 - n1 + 1)
    return r, c


def _reshape_array_as_excel(value, base_shape):
    try:
        return np.reshape(value, base_shape)
    except ValueError:
        res, (r, c) = np.empty(base_shape, object), value.shape
        res[:, :] = getattr(value, '_default', Error.errors['#N/A'])
        r = None if r == 1 else r
        c = None if c == 1 else c
        try:
            res[:r, :c] = value
        except ValueError:
            res[:, :] = Error.errors['#VALUE!']
    return res


class Ranges(object):
    input_fields = ('excel', 'sheet', 'n1', 'n2', 'r1', 'r2')

    def __init__(self, ranges=(), values=None, is_set=False, all_values=True):
        self.ranges = ranges
        self.values = values or {}
        self.is_set = is_set
        self.all_values = all_values or not ranges
        self._value = sh.NONE

    def pushes(self, refs, values=(), context=None):
        for r, v in itertools.zip_longest(refs, values, fillvalue=sh.EMPTY):
            self.push(r, value=v, context=context)
        self.is_set = self.is_set or len(self.ranges) > 1
        return self

    @staticmethod
    def format_range(*args, **kwargs):
        return range2parts(*args, **kwargs)

    def set_value(self, rng, value=sh.EMPTY):
        self._value = sh.NONE
        self.ranges += rng,
        if value is not sh.EMPTY:
            if not isinstance(value, Array):
                if not np.ndim(value):
                    value = [[value]]
                value = np.asarray(value, object)
            shape = _shape(**rng)
            value = _reshape_array_as_excel(value, shape)
            self.values[rng['name']] = (rng, value)
        else:
            self.all_values = False
        return self

    @staticmethod
    def get_range(format_range, ref, context=None):
        context = context or {}
        m = _re_range.match(ref).groupdict().items()
        m = {k: v for k, v in m if v is not None}
        if 'ref' in m:
            raise ValueError
        i = sh.combine_dicts(context, m)
        return dict(format_range(('name', 'n1', 'n2'), **i))

    def push(self, ref, value=sh.EMPTY, context=None):
        rng = self.get_range(self.format_range, ref, context)
        return self.set_value(rng, value)

    def __add__(self, other):  # Expand.
        ranges = self.ranges[1:] + other.ranges
        rng = sh.selector(self.input_fields, self.ranges[0])
        for k in ('r1', 'r2'):
            rng[k] = int(rng[k])

        for r in ranges:
            if not _has_same_sheet(rng, r):
                raise RangeValueError('{}:{}'.format(self, other))
            else:
                rng['r1'] = min(rng['r1'], int(r['r1']))
                rng['n1'] = min(rng['n1'], r['n1'])
                rng['r2'] = max(rng['r2'], int(r['r2']))
                rng['n2'] = max(rng['n2'], r['n2'])

        rng = dict(self.format_range(('name', 'n1', 'n2'), **rng))
        all_values, values = self.all_values and other.all_values, None
        if all_values:
            values = sh.combine_dicts(self.values, other.values)
            value = _assemble_values(rng, values)
            return Ranges().push(rng['name'], value)
        return Ranges((rng,), all_values=False)

    def __or__(self, other):  # Union.
        base = self.ranges
        for r0 in other.ranges:
            stack = [r0]
            for b in base:
                s = stack.copy()
                stack = []
                for r in s:
                    stack.extend(_split(b, r, format_range=self.format_range))
            base += tuple(stack)
        values = sh.combine_dicts(self.values, other.values)
        return Ranges(base, values, True, self.all_values and other.all_values)

    def __and__(self, other):  # Intersection.
        r = []
        for rng in other.ranges:
            r.extend(_intersect(
                rng, self.ranges, format_range=self.format_range
            ))
        values = sh.combine_dicts(self.values, other.values)
        is_set = self.is_set or other.is_set
        return Ranges(r, values, is_set, self.all_values and other.all_values)

    def __sub__(self, other):
        base = other.ranges
        for r0 in self.ranges:
            stack = [r0]
            for b in base:
                s = stack.copy()
                stack = []
                for r in s:
                    stack.extend(_split(b, r, format_range=self.format_range))
            base += tuple(stack)
        base, values = base[len(other.ranges):], self.values
        return Ranges(base, values, True, self.all_values)

    def simplify(self):
        rng = self.ranges
        if len(rng) <= 1:
            return self
        it = range(min(r['n1'] for r in rng), max(r['n2'] for r in rng) + 1)
        it = ['{0}:{0}'.format(_index2col(c)) for c in it]
        spl = (self & Ranges(is_set=False).pushes(it))._merge()
        spl.all_values = self.all_values
        return spl

    def _merge(self):
        # noinspection PyPep8
        key = lambda x: (x['n1'], int(x['r1']), -x['n2'], -int(x['r2']))
        rng = self.ranges
        for merge, select in ((_merge_raw_update, 1), (_merge_col_update, 0)):
            it, rng = sorted(rng, key=key), []
            for r in it:
                if not (rng and merge(rng[-1], r)):
                    if rng:
                        rng.append(rng.pop())
                    if select:
                        r = sh.selector(self.input_fields, r)
                    rng.append(r)
        rng = [dict(self.format_range(['name'], **r)) for r in rng]
        return Ranges(tuple(rng), self.values, self.is_set, self.all_values)

    def __repr__(self):
        ranges = ', '.join(r['name'] for r in self.ranges)
        value = '={}'.format(self.value) if ranges and self.all_values else ''
        return '<%s>(%s)%s' % (self.__class__.__name__, ranges, value)

    @property
    def value(self):
        if self._value is not sh.NONE:
            return self._value
        if not self.all_values:
            raise RangeValueError(str(self))
        stack, values = list(self.ranges), []
        while stack:
            update = False
            for k, (rng, value) in sorted(self.values.items()):
                if not stack:
                    break
                i = {}
                new_rng = _split(
                    rng, stack[-1], intersect=i, format_range=self.format_range
                )
                if i:
                    update = True
                    stack.pop()
                    stack.extend(new_rng)
                    r, c = _get_indices_intersection(rng, i)
                    values.append(value[:, c][r])
            else:
                if not update:
                    break

        if self.is_set:
            self._value = np.concatenate([v.ravel() for v in values])
        elif values:
            self._value = values[0]
        else:
            self._value = np.asarray([[Error.errors['#NULL!']]], object)
        return self._value
