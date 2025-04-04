#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2025 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
It provides Operand classes.
"""

# noinspection PyCompatibility
import regex
import functools
import schedula as sh
from . import Token
from ..errors import TokenError
from .parenthesis import _update_n_args

maxcol = 16384
maxrow = 1048576


class XlError(sh.Token):
    pass


NULL = XlError('#NULL!')
DIV = XlError('#DIV/0!')
VALUE = XlError('#VALUE!')
REF = XlError('#REF!')
NUM = XlError('#NUM!')
NAME = XlError('#NAME?')
NA = XlError('#N/A')


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

    def set_expr(self, *tokens):
        self.attr['expr'] = '"%s"' % self.name


class Empty(Operand):
    # noinspection PyMissingConstructor
    def __init__(self):
        self.source, self.attr = None, {'name': ''}

    @staticmethod
    def compile():
        return 0


_re_error = regex.compile(r'''
    ^\s*(?>
        (?>
            '(\[(?>[^\[\]]+)\])?
            (?>(?>''|[^\?!*\/\[\]':"])+)?'
        |
            (\[(?>[0-9]+)\])(?>(?>''|[^\?!*\/\[\]':"])+)?
        |
            (?>[^\W\d][\w\.]*)
        |
            '(?>(?>''|[^\?!*\/\[\]':"])+)'
        )!
    )?(?P<name>\#(?>NULL!|DIV/0!|VALUE!|REF!|NUM!|NAME\?|N/A))\s*
''', regex.IGNORECASE | regex.X | regex.DOTALL)


class Error(Operand):
    _re = _re_error
    errors = {str(k): k for k in (NULL, DIV, VALUE, REF, NUM, NAME, NA)}

    def compile(self):
        return self.errors[self.name]


class Number(Operand):
    _re = regex.compile(
        r'^\s*(?P<name>(?>[0-9]+(?>\.[0-9]+)?|\.[0-9]+)(?>E[+-][0-9]+)?|'
        r'TRUE(?!\(\))|FALSE(?!\(\)))(?!([a-z]|[0-9]|\.|\s*\:))\s*',
        regex.IGNORECASE
    )

    def compile(self):
        return eval(self.name.capitalize())


_re_ref = r'(?P<ref>[[:alpha:]_\\]+[[:alnum:]\.\_\\]*)'
_re_sheet_id = r"""
    (?>
        '((?P<directory>[^\[]+)?\/?\[(?P<filename>[^\[\]]+)\])?
         (?P<sheet>(?>''|[^\?*\/\[\]':\\])+)?'
    |
        (\[(?P<excel_id>[0-9]+)\])(?P<sheet>(?>''|[^\?!*\/\[\]':"])+)?
    |
        (?P<sheet>[^\W\d][\w\.]*)
    |
        '(?P<sheet>(?>''|[^\?*\/\[\]':\\])+)'
    )
"""
_re_range = r"""
    (?>
        (?>
            (?>
                R\[(?P<rr1>[\+-]?[1-9]\d*)\]C\[(?P<rc1>[\+-]?[1-9]\d*)\]
                (?>:R\[(?P<rr2>[\+-]?[1-9]\d*)\]C\[(?P<rc2>[\+-]?[1-9]\d*)\])?
            )
        |
            R\[(?P<rr1>[\+-]?[1-9]\d*)\]C\[(?P<rc1>[\+-]?[1-9]\d*)\](?P<anchor>\#)?
        |
            R\[(?P<rr1>[\+-]?[1-9]\d*)\]:R\[(?P<rr2>[\+-]?[1-9]\d*)\]
        |
            C\[(?P<rc1>[\+-]?[1-9]\d*)\]:C\[(?P<rc2>[\+-]?[1-9]\d*)\]
        )
    |
        (?>
            %s!
        )?
        (?>
            (?>
                (?>
                    \$?(?P<c1>[A-Z]{1,3})?\$?(?P<r1>[1-9]\d*)?
                    (?>:\$?(?P<c2>[A-Z]{1,3}))(\$?(?P<r2>[1-9]\d*))?
                )
            |
                \$?(?P<c1>[A-Z]{1,3})\$?(?P<r1>[1-9]\d*)(?P<anchor>\#)?
            |
                \$?(?P<c1>[A-Z]{1,3}):\$?(?P<c2>[A-Z]{1,3})
            |
                \$?(?P<r1>[1-9]\d*):\$?(?P<r2>[1-9]\d*)
            )(?![_\.\w])
        |
            (?>
                (?>
                    R(?P<r1>[1-9]\d*)C(?P<n1>[1-9]\d*)
                    (?>:R(?P<r2>[1-9]\d*)C(?P<n2>[1-9]\d*))?
                )
            |
                R(?P<r1>[1-9]\d*)C(?P<n1>[1-9]\d*)(?P<anchor>\#)?
            |
                R(?P<r1>[1-9]\d*):R(?P<r2>[1-9]\d*)
            |
                C(?P<n1>[1-9]\d*):C(?P<n2>[1-9]\d*)
            )(?![_\.\w])
        |
            %s
        )
    )
    (?![\(\w])
""" % (_re_sheet_id, _re_ref)
_re_range = regex.compile(
    r'^(?>(?P<anchor>(\_[Xx][Ll][Ff][Nn]\.)?ANCHORARRAY\({0}?\))|'
    r'(?P<indirect>INDIRECT\("{0}?"\))|{0})'.format(
        _re_range
    ), regex.IGNORECASE | regex.X | regex.DOTALL
)
_re_ref = regex.compile(
    r'^(?>{0}!)?{1}'.format(_re_sheet_id, _re_ref),
    regex.IGNORECASE | regex.X | regex.DOTALL
)
_re_sheet_id = regex.compile(
    r'^{0}'.format(_re_sheet_id), regex.IGNORECASE | regex.X | regex.DOTALL
)


@functools.lru_cache()
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
    return _index2col(maxcol)


@functools.lru_cache(None)
def _maxrow():
    return str(maxrow)


def _build_cel(c, r):
    r = str(r and int(r) or '')
    return c != _maxcol() and c or '', r != _maxrow() and r or ''


def _build_ref(c1, r1, c2, r2, anchor=''):
    (c1, r1), v2 = _build_cel(c1, r1), '{}{}'.format(*_build_cel(c2, r2))
    v1 = '{}{}{}'.format(c1, r1, anchor)
    if v1 == v2 and c1 and r1:
        if v1:
            return v1
        raise ValueError
    if anchor:
        raise ValueError
    return '%s:%s' % (v1, v2)


_re_build_id = regex.compile(r'^[0-9]+$')


def _build_sheet_id(sheet='', directory='', filename='', **kw):
    sheet = sheet.replace("''", "'").upper()
    if filename:
        if _re_build_id.match(filename):
            sheet = "[%s]%s" % (filename, sheet)
        else:
            if directory and not directory.endswith('/'):
                directory += '/'
            sheet = "'%s[%s]%s'" % (directory, filename, sheet)
    elif ' ' in sheet:
        sheet = "'%s'" % sheet
    return sheet


def _build_id(ref, sheet_id):
    if sheet_id:
        return '{}!{}'.format(sheet_id, ref)
    return ref


def _sum(*args):
    return sum(map(float, args))


@functools.lru_cache(None)
def _range2parts(inputs, outputs):
    dsp = sh.Dispatcher(raises=True)
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
    dsp.add_data(data_id='ref', filters=(str.upper,))
    dsp.add_data(data_id='n1', default_value=0, initial_dist=100)
    dsp.add_data(data_id='r1', default_value='0', initial_dist=100)
    dsp.add_data(data_id='c2', default_value=_maxcol(), initial_dist=100)
    dsp.add_data(data_id='r2', default_value=_maxrow(), initial_dist=100)
    dsp.add_data(data_id='anchor', default_value='', initial_dist=100)
    dsp.add_function(
        None, _build_ref, ['c1', 'r1', 'c2', 'r2', 'anchor'], ['ref']
    )
    dsp.add_function(None, _build_id, ['ref', 'sheet_id'], ['name'])
    func = sh.DispatchPipe(dsp, '', inputs, outputs)
    func.output_type = 'all'
    return func


_keys = {
    'r1', 'r2', 'c1', 'c2', 'n1', 'n2', 'ref', 'name', 'sheet_id', 'anchor'
}


def fast_range2parts(**kw):
    inputs = {k: kw[k] for k in _keys if k in kw}

    for func in (fast_range2parts_v1, fast_range2parts_v2, fast_range2parts_v3,
                 fast_range2parts_v4, fast_range2parts_v5):
        try:
            parts = func(**inputs)
            parts.update(kw)
            return parts
        except TypeError:
            pass
    else:
        raise ValueError


def fast_range2parts_v1(r1, c1, sheet_id, anchor=''):
    n1 = _col2index(c1)
    ref = '{}{}{}'.format(*_build_cel(c1, r1), anchor).upper()
    return {
        'r1': r1, 'r2': r1, 'c1': c1, 'c2': c1, 'n1': n1, 'n2': n1, 'ref': ref,
        'name': _build_id(ref, sheet_id), 'anchor': anchor
    }


def fast_range2parts_v2(r1, c1, r2, c2, sheet_id):
    ref = _build_ref(c1, r1, c2, r2).upper()
    return {
        'r1': r1, 'r2': r2, 'c1': c1, 'c2': c2, 'n1': _col2index(c1),
        'n2': _col2index(c2), 'ref': ref, 'name': _build_id(ref, sheet_id)
    }


def fast_range2parts_v3(r1, n1, sheet_id, anchor=''):
    c1 = _index2col(n1)
    ref = '{}{}{}'.format(*_build_cel(c1, r1), anchor).upper()
    return {
        'r1': r1, 'r2': r1, 'c1': c1, 'c2': c1, 'n1': n1, 'n2': n1, 'ref': ref,
        'name': _build_id(ref, sheet_id), 'anchor': anchor
    }


def fast_range2parts_v4(r1, n1, r2, n2, sheet_id):
    c1, c2 = _index2col(n1), _index2col(n2)
    ref = _build_ref(c1, r1, c2, r2).upper()
    return {
        'r1': r1, 'r2': r2, 'c1': c1, 'c2': c2, 'n1': n1, 'n2': n2, 'ref': ref,
        'name': _build_id(ref, sheet_id)
    }


# noinspection PyUnusedLocal
def fast_range2parts_v5(ref, sheet_id):
    ref = ref.upper()
    return {'ref': ref, 'name': _build_id(ref, sheet_id)}


def range2parts(outputs, **inputs):
    excel_id = inputs.get('excel_id')
    if excel_id and excel_id != '0':
        inputs['directory'], inputs['filename'] = inputs.get(
            'external_links', {}
        ).get(excel_id, ('', excel_id))

    if 'sheet_id' not in inputs:
        inputs['sheet_id'] = _build_sheet_id(**inputs)
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
        ctx = (context or {}).copy()
        ctx.update(d)
        if ctx.get('anchor'):
            ctx['anchor'] = '#'
            ctx['is_ranges'] = False
            self.attr['is_reference'] = True
        if 'ref' in d:
            ctx.pop('sheet', None)
            self.attr['is_reference'] = True

        return range2parts(None, **ctx)

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
