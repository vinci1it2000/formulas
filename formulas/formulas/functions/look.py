# !/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
Python equivalents of logical excel functions.
"""
import regex
import functools
import numpy as np
from . import wrap_ufunc, Error, flatten, get_error, XlError, FoundError

FUNCTIONS = {}


def _get_type_id(obj):
    if isinstance(obj, bool):
        return 2
    elif isinstance(obj, (int, float)):
        return 0
    elif not isinstance(obj, XlError) and isinstance(obj, str):
        return 1


def _yield_vals(type_id, array):
    for i, v in enumerate(array, 1):
        if type_id == _get_type_id(v):
            yield i, v


def xmatch(lookup_value, lookup_array, match_type=1):
    res = [Error.errors['#N/A']]
    t_id = _get_type_id(lookup_value)
    if match_type > 0:
        def check(i, x, val, res):
            if x <= val:
                res[0] = i
                return x == val and i > 1
            return i > 1

    elif match_type < 0:
        def check(i, x, val, res):
            if x < val:
                return True
            res[0] = i
            return v == val

    else:
        t_id = _get_type_id(lookup_value)
        if t_id == 1:
            def sub(match):
                return {'\\': '', '?': '.', '*': '.*'}[match.groups()[0]]

            match = regex.compile(r'^%s$' % regex.sub(
                r'(?<!\\\~)\\(?P<sub>[\*\?])|(?P<sub>\\)\~(?=\\[\*\?])', sub,
                regex.escape(lookup_value)
            ), regex.IGNORECASE).match
        else:
            match = lambda x: x == lookup_value

        def check(i, x, val, res):
            if match(x):
                res[0] = i

    convert = lambda x: x
    if t_id == 1:
        convert = lambda x: x.upper()

    lookup_value = convert(lookup_value)
    for i, v in _yield_vals(t_id, lookup_array):
        if check(i, convert(v), lookup_value, res):
            break
    return res[0]


FUNCTIONS['MATCH'] = wrap_ufunc(
    xmatch,
    input_parser=lambda val, vec, match_type=1: (
        val, list(flatten(vec, None)), match_type
    ),
    check_error=lambda *a: get_error(a[:1]), excluded={1, 2}
)


def xlookup(lookup_val, lookup_vec, result_vec=None, match_type=1):
    """
    The vector form of LOOKUP looks in a one-row or one-column range (known as a
    vector) for a value and returns a value from the same position in a second
    one-row or one-column range.

    :param lookup_val:
        A value that LOOKUP searches for in the first vector.
    :type lookup_val: a number, text, a logical value, or a name or reference that refers to a value.

    :param lookup_vec: A range that contains only one row or one column.
    :param result_vec: A range that contains only one row or column.


    :type lookup_vec: an array containing text, numbers, or logical values.
    :type result_vec: must be the same size as lookup_vector.

    :return:
    """
    result_vec = lookup_vec if result_vec is None else result_vec
    r = xmatch(lookup_val, lookup_vec, match_type)
    if not isinstance(r, XlError):
        r = result_vec[r - 1]
    return r


FUNCTIONS['LOOKUP'] = wrap_ufunc(
    xlookup,
    input_parser=lambda val, vec, res=None: (
        val, list(flatten(vec, None)),
        res if res is None else list(flatten(res, None))
    ),
    check_error=lambda *a: get_error(a[:1]), excluded={1, 2}
)


def _hlookup_parser(val, vec, index, match_type=1, transpose=False):
    index = list(flatten(index, None))[0] - 1
    vec = np.matrix(vec)
    if transpose:
        vec = vec.T
    try:
        ref = list(flatten(vec[index].A1, None))
    except IndexError:
        raise FoundError(err=Error.errors['#REF!'])
    vec = list(flatten(vec[0].A1, None))
    return val, vec, ref, bool(match_type)


FUNCTIONS['HLOOKUP'] = wrap_ufunc(
    xlookup, input_parser=_hlookup_parser,
    check_error=lambda *a: get_error(a[:1]), excluded={1, 2, 3}
)
FUNCTIONS['VLOOKUP'] = wrap_ufunc(
    xlookup, input_parser=functools.partial(_hlookup_parser, transpose=True),
    check_error=lambda *a: get_error(a[:1]), excluded={1, 2, 3}
)
