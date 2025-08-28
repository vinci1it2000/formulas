#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2025 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
It provides Ranges class.
"""
import itertools
import numpy as np
from .tokens.operand import (
    _re_range, range2parts, _index2col, maxrow, maxcol, Error
)
from .errors import (
    RangeValueError, InvalidRangeError, InvalidRangeName, AnchorRangeName
)
from .functions import Array, _init_reshape
import schedula as sh


def _has_same_sheet(x, y):
    return x.get('sheet_id', True) == y.get('sheet_id', False)


def _intersect(x, y):
    if _has_same_sheet(x, y):
        n1, n2 = max(y['n1'], x['n1']), min(y['n2'], x['n2'])
        if n1 <= n2:
            r1 = max(int(y['r1']), int(x['r1']))
            r2 = min(int(y['r2']), int(x['r2']))
            if r1 <= r2:
                return {
                    'sheet_id': x['sheet_id'], 'n1': n1, 'r1': str(r1),
                    'n2': n2, 'r2': str(r2)
                }
    return {}


def _split(base, rng, intersect=None, format_range=range2parts):
    z = _intersect(base, rng)
    if not z:
        return rng,

    if intersect is not None:
        intersect.update(z)

    ranges = []
    rng = {
        'sheet_id': rng['sheet_id'], 'n1': rng['n1'], 'n2': rng['n2'],
        'r1': rng['r1'], 'r2': rng['r2']
    }
    it = ('n1', 'n2', 1), ('n2', 'n1', -1), ('r1', 'r2', 1), ('r2', 'r1', -1)
    for i, j, n in it:
        if z[i] != rng[i]:
            r = rng.copy()
            r[j] = str(int(z[i]) - n) if j[0] == 'r' else z[i] - n
            r = dict(format_range(('name', 'n1', 'n2'), **r))
            ranges.append(r)
            rng[i] = z[i]

    return tuple(ranges)


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
    r, c = int(base['r1']) or 1, base['n1'] or 1
    r = slice((int(i['r1']) or 1) - r, (int(i['r2']) or 1) - r + 1)
    c = slice((i['n1'] or 1) - c, (i['n2'] or 1) - c + 1)
    return r, c


def _assemble_values(base, values, out=None):
    if out is None:
        out = np.empty(_shape(**base), object)
        out[:, :] = ''
    for rng, value in values.values():
        ist = _intersect(base, rng)
        if ist:
            br, bc = _get_indices_intersection(base, ist)
            rr, rc = _get_indices_intersection(rng, ist)
            out[br, bc] = value[rr, rc]
    return out


# noinspection PyUnusedLocal
def _shape(n1, n2, r1, r2, **kw):
    r1, r2 = int(r1), int(r2)
    r = maxrow if r1 == 0 and r2 == maxrow else (r2 - r1 + 1)
    c = maxcol if n1 == 0 and n2 == maxcol else (n2 - n1 + 1)
    return r, c


def _reshape_array_as_excel(value, base_shape):
    try:
        return np.reshape(value, base_shape)
    except ValueError:
        res, r, c, val = _init_reshape(base_shape, value)
        try:
            res[:r, :c] = val
        except ValueError:
            res[:, :] = Error.errors['#VALUE!']
    return res


class Ranges:
    __slots__ = 'ranges', 'values', '_value'

    def __init__(self, ranges=(), values=None):
        self.ranges = ranges
        self.values = values or {}
        self._value = sh.NONE

    def pushes(self, refs, values=(), context=None):
        for r, v in itertools.zip_longest(refs, values, fillvalue=sh.EMPTY):
            self.push(r, value=v, context=context)
        return self

    @staticmethod
    def format_range(*args, **kwargs):
        return range2parts(*args, **kwargs)

    def set_value(self, rng, value=sh.EMPTY):
        self._value = sh.NONE
        self.ranges += rng,
        if value is not sh.EMPTY:
            if isinstance(value, Ranges):
                value = value.value
            if not isinstance(value, Array):
                if not np.ndim(value):
                    value = [[value]]
                value = np.asarray(value, object)
            shape = _shape(**rng)
            value = _reshape_array_as_excel(value, shape)
            self.values[rng['name']] = (rng, value)

        return self

    @property
    def is_set(self):
        return len(self.ranges) > 1

    @staticmethod
    def get_range(ref, context=None, raise_anchor=True):
        ctx = (context or {}).copy()
        try:
            for k, v in _re_range.match(ref).groupdict().items():
                if v is None:
                    continue
                if k == 'ref':
                    raise InvalidRangeName
                if raise_anchor and k == 'anchor':
                    raise AnchorRangeName
                ctx[k] = v
        except AttributeError:
            raise InvalidRangeName
        return Ranges.format_range(('name', 'n1', 'n2'), **ctx)

    def push(self, ref, value=sh.EMPTY, context=None, raise_anchor=True):
        return self.set_value(self.get_range(ref, context, raise_anchor), value)

    def __add__(self, other):  # Expand.
        ranges = self.ranges[1:] + other.ranges
        rng = self.ranges[0]
        rng = {
            'sheet_id': rng['sheet_id'], 'n1': rng['n1'], 'n2': rng['n2'],
            'r1': rng['r1'], 'r2': rng['r2']
        }
        for k in ('r1', 'r2'):
            rng[k] = int(rng[k])

        for r in ranges:
            if not _has_same_sheet(rng, r):
                raise InvalidRangeError('{}:{}'.format(self, other))
            else:
                rng['r1'] = min(rng['r1'], int(r['r1']))
                rng['n1'] = min(rng['n1'], r['n1'])
                rng['r2'] = max(rng['r2'], int(r['r2']))
                rng['n2'] = max(rng['n2'], r['n2'])

        rng = self.format_range(('name', 'n1', 'n2'), **rng)
        if self.values and other.values:
            values = self.values.copy()
            values.update(other.values)
            value = _assemble_values(rng, values)
            return Ranges().push(rng['name'], value)
        return Ranges((rng,))

    def __or__(self, other):  # Union.
        values = self.values.copy()
        values.update(other.values)
        return Ranges(self.ranges + other.ranges, values)

    def intersect(self, other):
        for rng in other.ranges:
            for r in self.ranges:
                z = _intersect(rng, r)
                if z:
                    yield z

    def __and__(self, other):  # Intersection.
        r = tuple(self.format_range(('name', 'n1', 'n2'), **i)
                  for i in self.intersect(other))
        values = self.values.copy()
        values.update(other.values)
        return Ranges(r, values)

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
        return Ranges(base, values)

    def simplify(self):
        rng = self.ranges
        if len(rng) <= 1:
            return self
        it = range(min(r['n1'] for r in rng), max(r['n2'] for r in rng) + 1)
        it = ['{0}:{0}'.format(_index2col(c)) for c in it]
        spl = (self & Ranges().pushes(it))._merge()
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
                        r = {
                            'sheet_id': r['sheet_id'], 'n1': r['n1'],
                            'n2': r['n2'], 'r1': r['r1'], 'r2': r['r2']
                        }
                    rng.append(r)
        rng = [self.format_range(['name'], **r) for r in rng]
        return Ranges(tuple(rng), self.values)

    def __repr__(self):
        ranges = ', '.join(r['name'] for r in self.ranges)
        if ranges and self.values:
            value = '={}'.format(self.value)
            if 'np.' in value:  # Correct formatting for numpy v2.x.
                value = '={}'.format(np.array2string(
                    self.value, formatter={'all': '{}'.format}
                ))
        else:
            value = ''

        return '<%s>(%s)%s' % (self.__class__.__name__, ranges, value)

    @property
    def value(self):
        if self._value is not sh.NONE:
            return self._value
        if self.ranges and not self.values:
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
