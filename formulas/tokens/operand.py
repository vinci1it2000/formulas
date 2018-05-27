#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
It provides Operand classes.
"""

from . import Token
# noinspection PyCompatibility
import regex
import sys
import functools
import schedula as sh
from ..errors import TokenError
from .parenthesis import _update_n_args

maxsize = sys.maxsize


class XlError(sh.Token):
    pass


class Operand(Token):
    def ast(self, tokens, stack, builder):
        if tokens and isinstance(tokens[-1], Operand):
            raise TokenError()
        super(Operand, self).ast(tokens, stack, builder)
        builder.append(self)
        _update_n_args(stack)


class String(Operand):
    _re = regex.compile(r'^\s*"(?P<name>(?>""|[^"])*)"\s*')

    def compile(self):
        return self.name.replace('""', '"')


class Error(Operand):
    _re = regex.compile(
        r'^\s*(?P<name>\#(?>NULL!|DIV/0!|VALUE!|REF!|NUM!|NAME\?|N/A))\s*',
        regex.IGNORECASE
    )
    errors = {}
    for k in ('NULL!', 'DIV/0!', 'VALUE!', 'REF!', 'NUM!', 'NAME?', 'N/A'):
        k = '#%s' % k
        errors[k] = XlError(k)

    def compile(self):
        return self.errors[self.name]


class Number(Operand):
    _re = regex.compile(
        r'^\s*(?P<name>[0-9]+(?>\.[0-9]+)?(?>E[+-]?[0-9]+|%)?|TRUE|FALSE)\s*',
        regex.IGNORECASE
    )

    def compile(self):
        return eval(self.name.capitalize())


_re_range = r"""
    (?>
        (?>
            (?>
                '(\[(?P<excel>[^\[\]]+)\])?
                 (?P<sheet>(?>''|[^\?!*\/\[\]':"])+)?'
            |
                (\[(?P<excel_id>[0-9]+)\])(?P<sheet>[^\W\d][\w\.]*)
            |
                (?P<sheet>[^\W\d][\w\.]*)
            |
                '(?P<sheet>(?>''|[^\?!*\/\[\]':"])+)'
            )!
        )?
        (?>
            (?>
                (?>
                    \$?(?P<c1>[A-Z]+)?\$?(?P<r1>[1-9]\d*)?
                    (?>:\$?(?P<c2>[A-Z]+)?)(\$?(?P<r2>[1-9]\d*)?)?
                )
            |
                \$?(?P<c1>[A-Z]+)\$?(?P<r1>[1-9]\d*)
            |
                \$?(?P<c1>[A-Z]+):\$?(?P<c2>[A-Z]+)
            |
                \$?(?P<r1>[1-9]\d*):\$?(?P<r2>[1-9]\d*)
            )
        |
            (?>
                (?>
                    R(?P<r1>[1-9]\d*)C(?P<n1>[1-9]\d*)
                    (?>:R(?P<r2>[1-9]\d*)C(?P<n2>[1-9]\d*))?
                )
            |
                R(?P<r1>[1-9]\d*)C(?P<n1>[1-9]\d*)
            |
                R(?P<r1>[1-9]\d*):R(?P<r2>[1-9]\d*)
            |
                C(?P<n1>[1-9]\d*):C(?P<n2>[1-9]\d*)
            )
        |
            (?P<ref>[A-Z_\\]+[A-Z0-9\.\_]*)
        )
    |
        (?>
            (?>
                R\[(?P<rr1>[\+-]?[1-9]\d*)\]C\[(?P<rc1>[\+-]?[1-9]\d*)\]
                (?>:R\[(?P<rr2>[\+-]?[1-9]\d*)\]C\[(?P<rc2>[\+-]?[1-9]\d*)\])?
            )
        |
            R\[(?P<rr1>[\+-]?[1-9]\d*)\]C\[(?P<rc1>[\+-]?[1-9]\d*)\]
        |
            R\[(?P<rr1>[\+-]?[1-9]\d*)\]:R\[(?P<rr2>[\+-]?[1-9]\d*)\]
        |
            C\[(?P<rc1>[\+-]?[1-9]\d*)\]:C\[(?P<rc2>[\+-]?[1-9]\d*)\]
        )
    )
    (?!\()
    """
_re_range = regex.compile(
    r'^(?>(?P<indirect>INDIRECT\("{0}?"\))|{0})'.format(_re_range),
    regex.IGNORECASE | regex.X | regex.DOTALL
)


def _index2col(index):
    index = int(index) - 1
    if index < 0:
        return ''
    d = index // 26
    chr1 = _index2col(d) if d > 0 else ''
    return '%s%s' % (chr1, chr(ord('A') + index % 26))


def _col2index(col):
    index, power = 0, 1
    for ch in col.upper()[::-1]:
        index += (ord(ch) - ord('A') + 1) * power
        power *= 26
    return index


@functools.lru_cache(None)
def _maxcol():
    return _index2col(maxsize)


@functools.lru_cache(None)
def _maxrow():
    return str(maxsize)


def _build_cel(c, r):
    r = str(r and int(r) or '')
    return c != _maxcol() and c or '', r != _maxrow() and r or ''


def _build_ref(c1, r1, c2, r2):
    (c1, r1), v2 = _build_cel(c1, r1), '{}{}'.format(*_build_cel(c2, r2))
    v1 = '{}{}'.format(c1, r1)
    if v1 == v2 and c1 and r1:
        if v1:
            return v1
        raise ValueError()
    return '%s:%s' % (v1, v2)

_re_build_id = regex.compile(r'^[0-9]+$')
def _build_id(ref, sheet='', excel=''):
    if excel:
        sheet = "[%s]%s" % (excel, sheet.replace("''", "'"))
        if not _re_build_id.match(excel):
            sheet = "'%s'" % sheet

    return '!'.join(s for s in (sheet, ref) if s)


def _sum(*args):
    return sum(map(float, args))


@functools.lru_cache(None)
def _range2parts(inputs, outputs):
    dsp = sh.Dispatcher()
    dsp.add_data(data_id='cr', default_value='1')
    dsp.add_data(data_id='cc', default_value='1')
    dsp.add_function('relative2absolute', _sum, ['cr', 'rr1'], ['r1'])
    dsp.add_function('relative2absolute', _sum, ['cc', 'rc1'], ['n1'])
    dsp.add_function('relative2absolute', _sum, ['cr', 'rr2'], ['r2'])
    dsp.add_function('relative2absolute', _sum, ['cc', 'rc2'], ['n2'])
    dsp.add_function(function=_index2col, inputs=['n1'], outputs=['c1'])
    dsp.add_function(function=_index2col, inputs=['n2'], outputs=['c2'])
    dsp.add_function(function=_col2index, inputs=['c1'], outputs=['n1'])
    dsp.add_function(function=_col2index, inputs=['c2'], outputs=['n2'])
    dsp.add_function(function=sh.bypass, inputs=['c1'], outputs=['c2'])
    dsp.add_function(function=sh.bypass, inputs=['r1'], outputs=['r2'])
    dsp.add_function(function=lambda x, y: x[y],
                     inputs=['external_links', 'excel_id'], outputs=['excel'])
    dsp.add_function(function=sh.bypass, weight=1,
                     inputs=['excel_id'], outputs=['excel'])
    dsp.add_data(data_id='excel', filters=(str.upper,))
    dsp.add_data(data_id='sheet', default_value='', filters=(str.upper,))
    dsp.add_data(data_id='ref', filters=(str.upper,))
    dsp.add_data(data_id='name', filters=(str.upper,))
    dsp.add_data(data_id='n1', default_value=0, initial_dist=100)
    dsp.add_data(data_id='r1', default_value='0', initial_dist=100)
    dsp.add_data(data_id='n2', default_value=maxsize, initial_dist=100)
    dsp.add_data(data_id='r2', default_value='%s' % maxsize, initial_dist=100)
    dsp.add_function(None, _build_ref, ['c1', 'r1', 'c2', 'r2'], ['ref'])
    dsp.add_function(None, _build_id, ['ref', 'sheet', 'excel'], ['name'])
    func = sh.DispatchPipe(dsp, '', inputs, outputs)
    func.output_type = 'all'
    return func


_keys = {'r1', 'r2', 'excel', 'c1', 'c2', 'n1', 'n2', 'sheet', 'ref', 'name'}


def fast_range2parts(**kw):
    inputs = sh.selector(_keys, kw, allow_miss=True)

    for func in (fast_range2parts_v1, fast_range2parts_v2, fast_range2parts_v3):
        try:
            return sh.combine_dicts(kw, base=func(**inputs))
        except TypeError:
            pass
    else:
        raise ValueError


def fast_range2parts_v1(r1, c1, excel, sheet=''):
    n1, sheet = _col2index(c1), sheet.upper()
    ref, excel = '{}{}'.format(*_build_cel(c1, r1)).upper(), excel.upper()
    return {
        'r1': r1, 'r2': r1, 'c1': c1, 'c2': c1, 'n1': n1, 'n2': n1, 'ref': ref,
        'sheet': sheet, 'name': _build_id(ref, sheet, excel), 'excel': excel
    }


def fast_range2parts_v2(r1, c1, r2, c2, excel, sheet=''):
    sheet = sheet.upper()
    ref, excel = _build_ref(c1, r1, c2, r2).upper(), excel.upper()
    return {
        'r1': r1, 'r2': r2, 'c1': c1, 'c2': c2, 'n1': _col2index(c1),
        'n2': _col2index(c2), 'ref': ref, 'sheet': sheet,
        'name': _build_id(ref, sheet, excel), 'excel': excel
    }


def fast_range2parts_v3(r1, n1, r2, n2, excel, sheet=''):
    sheet = sheet.upper()
    c1, c2 = _index2col(n1), _index2col(n2)
    ref, excel = _build_ref(c1, r1, c2, r2).upper(), excel.upper()
    return {
        'r1': r1, 'r2': r2, 'c1': c1, 'c2': c2, 'n1': n1, 'n2': n2, 'ref': ref,
        'sheet': sheet, 'name': _build_id(ref, sheet, excel), 'excel': excel
    }


def range2parts(outputs, **inputs):
    if 'excel_id' in inputs:
        inputs.pop('excel', None)
    elif 'excel' not in inputs:
        inputs['excel'] = ''
    try:
        return fast_range2parts(**inputs)
    except ValueError:
        keys = tuple(sorted(inputs))
        outputs = outputs and tuple(outputs) or outputs
        return dict(_range2parts(keys, outputs)(*(inputs[k] for k in keys)))


class Range(Operand):
    _re = _re_range

    def process(self, match, context=None):
        d = super(Range, self).process(match)
        if len(d) <= 1 and 'indirect' not in d and 'ref' in d:
            try:
                from .function import Function
                if Function(self.source).name == d['ref']:
                    return {}
            except TokenError:
                pass
        if 'ref' in d:
            self.attr['is_reference'] = True
        return range2parts(None, **sh.combine_dicts(context or {}, d))

    def __repr__(self):
        if self.attr.get('is_ranges', False):
            from ..ranges import Ranges
            return '{} <{}>'.format(self.name, Ranges.__name__)
        return super(Range, self).__repr__()

    def compile(self):
        if self.attr.get('is_ranges', False):
            from ..ranges import Ranges
            return Ranges().push(self.attr['name'])
        return sh.EMPTY
