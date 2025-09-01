# !/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2025 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
Python equivalents of lookup and reference Excel functions.
"""
import math
import regex
import functools
import collections
import numpy as np
import schedula as sh
from . import (
    wrap_func, wrap_ufunc, Error, get_error, XlError, FoundError, Array,
    parse_ranges, _text2num, replace_empty, raise_errors, COMPILING,
    wrap_impure_func, flatten, _convert2float, return_2d_func
)
from ..ranges import Ranges
from ..cell import CELL

FUNCTIONS = {}


def _get_type_id(obj):
    if isinstance(obj, (bool, np.bool_)):
        return 2
    elif isinstance(obj, (str, np.str_)) and not isinstance(obj, XlError):
        return 1
    return 0


def _xref(func, cell=None, ref=None):
    try:
        return func((ref or cell).ranges[0]).view(Array)
    except IndexError:
        return Error.errors['#NULL!']


def xrow(cell=None, ref=None):
    return _xref(
        lambda r: np.arange(int(r['r1']), int(r['r2']) + 1)[:, None], cell, ref
    )


def xcolumn(cell=None, ref=None):
    return _xref(lambda r: np.arange(r['n1'], r['n2'] + 1)[None, :], cell, ref)


FUNCTIONS['COLUMN'] = {
    'extra_inputs': collections.OrderedDict([(CELL, None)]),
    'function': wrap_func(xcolumn, ranges=True)
}
FUNCTIONS['ROW'] = {
    'extra_inputs': collections.OrderedDict([(CELL, None)]),
    'function': wrap_func(xrow, ranges=True)
}


def xaddress(row_num, column_num, abs_num=1, a1=True, sheet_text=None):
    from ..tokens.operand import _index2col
    column_num, row_num = int(column_num), int(row_num)
    if column_num <= 0 or row_num <= 0:
        return Error.errors['#VALUE!']
    if a1 is sh.EMPTY or not int(a1):
        m = {1: 'R{1}C{0}', 2: 'R{1}C[{0}]', 3: 'R[{1}]C{0}', 4: 'R[{1}]C[{0}]'}
    else:
        column_num = _index2col(column_num)
        m = {1: '${}${}', 2: '{}${}', 3: '${}{}', 4: '{}{}'}
    address = m[int(abs_num)].format(column_num, row_num)
    if sheet_text:
        if sheet_text is sh.EMPTY:
            return "!{}".format(address)
        address = "'{}'!{}".format(str(sheet_text).replace("'", "''"), address)
    return address


FUNCTIONS['ADDRESS'] = wrap_ufunc(
    xaddress, input_parser=lambda *a: a, args_parser=lambda *a: a
)


def xareas(ref):
    return len(ref.ranges) or Error.errors['#NULL!']


def xcolumns(ref, *, axis=1):
    r = ref.ranges
    n = len(r)
    if n == 1:
        if axis:
            return r[0]['n2'] - r[0]['n1'] + 1
        return int(r[0]['r2']) - int(r[0]['r1']) + 1
    if n == 0:
        return Error.errors['#NULL!']
    if n > 1:
        return Error.errors['#REF!']


FUNCTIONS['AREAS'] = wrap_func(xareas, ranges=True)
FUNCTIONS['COLUMNS'] = wrap_func(xcolumns, ranges=True)
FUNCTIONS['ROWS'] = wrap_func(functools.partial(xcolumns, axis=0), ranges=True)

FUNCTIONS['CHOOSE'] = wrap_ufunc(
    np.choose,
    check_error=lambda i, *a: get_error(i),
    input_parser=lambda i, *a: (
        int(_convert2float(i)) - 1, np.array(a, object)
    ),
    args_parser=lambda i, a1, *a: (replace_empty(i), a1) + a
)


def xchoosecols(arr, col, *cols, axis=1):
    raise_errors(col, *cols)
    arr = np.atleast_2d(arr)
    indices = []
    for v in (col, *cols):
        v = np.atleast_2d(replace_empty(v))
        if 1 not in v.shape:
            return Error.errors['#VALUE!']
        indices.append(v)

    indices = np.array(tuple(map(_convert2float, flatten(indices, None))), int)
    if (indices == 0).any():
        return Error.errors['#VALUE!']
    indices[indices > 0] -= 1
    return np.take(arr, indices, axis=axis).view(Array)


FUNCTIONS['_XLFN.CHOOSECOLS'] = FUNCTIONS['CHOOSECOLS'] = wrap_func(xchoosecols)
FUNCTIONS['_XLFN.CHOOSEROWS'] = FUNCTIONS['CHOOSEROWS'] = wrap_func(
    functools.partial(xchoosecols, axis=0)
)


def xtocol(array, ignore=0, scan_by_column=0, axis=1):
    array = np.atleast_2d(array)
    if scan_by_column:
        array = array.T
    if ignore == 0:
        check = None
    elif ignore == 1:
        check = lambda x: x is not sh.EMPTY
    elif ignore == 2:
        check = lambda x: not get_error(x)
    elif ignore == 3:
        check = lambda x: not get_error(x) and x is not sh.EMPTY
    else:
        return Error.errors['#VALUE!']

    val = np.array([list(flatten(array, check))], object)
    if axis:
        val = val.T
    return val.tolist()


FUNCTIONS['_XLFN.TOCOL'] = FUNCTIONS['TOCOL'] = wrap_ufunc(
    xtocol,
    excluded={0},
    check_nan=False,
    check_error=lambda array, *a: get_error(*a),
    input_parser=lambda array, ignore=0, scan_by_column=0: (
        array, int(_convert2float(ignore)), int(scan_by_column)
    ),
    args_parser=lambda array, ignore=0, scan_by_column=0: (
        array, replace_empty(ignore), replace_empty(scan_by_column)
    ),
    return_func=return_2d_func
)
FUNCTIONS['_XLFN.TOROW'] = FUNCTIONS['TOROW'] = wrap_ufunc(
    functools.partial(xtocol, axis=0),
    excluded={0},
    check_nan=False,
    check_error=lambda array, *a: get_error(*a),
    input_parser=lambda array, ignore=0, scan_by_column=0: (
        array, int(_convert2float(ignore)), int(scan_by_column)
    ),
    args_parser=lambda array, ignore=0, scan_by_column=0: (
        array, replace_empty(ignore), replace_empty(scan_by_column)
    ),
    return_func=return_2d_func
)


def xsingle(cell, rng):
    if not isinstance(rng, Ranges):
        if isinstance(rng, Array):
            return rng.ravel()[0]
        return rng
    if len(rng.ranges) == 1 and not rng.is_set and rng.value.shape[1] == 1:
        rng = rng & Ranges((sh.combine_dicts(
            rng.ranges[0], sh.selector(('r1', 'r2'), cell.ranges[0])
        ),))
        if rng.ranges:
            return rng
    return Error.errors['#VALUE!']


FUNCTIONS['_XLFN.SINGLE'] = FUNCTIONS['SINGLE'] = {
    'extra_inputs': collections.OrderedDict([(COMPILING, False), (CELL, None)]),
    'function': wrap_impure_func(wrap_func(xsingle, ranges=True))
}


def _index(arrays, row_num, col_num, area_num, is_reference, is_array):
    err = get_error(row_num, col_num, area_num)
    if err:
        return err
    area_num = int(area_num) - 1
    if area_num < 0:
        return Error.errors['#VALUE!']
    try:
        array = arrays[area_num]

        if col_num is None:
            col_num = 1
            if 1 in array.shape:
                if array.shape[0] == 1:
                    row_num, col_num = col_num, row_num
            elif is_reference:
                array = None
            elif not is_array:
                col_num = None

        if row_num is not None:
            row_num = int(row_num) - 1
            if row_num < -1:
                return Error.errors['#VALUE!']
            row_num = max(0, row_num)

        if col_num is not None:
            col_num = int(col_num) - 1
            if col_num < -1:
                return Error.errors['#VALUE!']
            col_num = max(0, col_num)

        val = array[row_num, col_num]
        return 0 if val is sh.EMPTY else val
    except (IndexError, TypeError):
        return Error.errors['#REF!']


def xindex(array, row_num, col_num=None, area_num=1):
    is_reference = isinstance(array, Ranges)
    if is_reference:
        arrays = [Ranges((rng,), array.values).value for rng in array.ranges]
    else:
        arrays = [array]

    row_num, col_num, area_num = parse_ranges(row_num, col_num, area_num)[0]

    res = np.vectorize(_index, excluded={0}, otypes=[object])(
        arrays, row_num, col_num, area_num, is_reference,
        isinstance(row_num, np.ndarray)
    )
    if not res.shape:
        res = res.reshape(1, 1)
        if isinstance(res[0, 0], np.ndarray):
            res = res[0, 0]
    return res.view(Array)


FUNCTIONS['INDEX'] = wrap_func(xindex, ranges=True)


def _binary_search(index, arr, target, eq, asc=True):
    left = 0
    right = len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        val = arr[mid]

        if eq(index[mid], val, True):
            break

        if val < target:
            left = mid + 1 if asc else left
            right = right if asc else mid - 1
        else:
            left = left if asc else mid + 1
            right = mid - 1 if asc else right


def xmatch(
        lookup_value_type, lookup_value, lookup_value_raw, lookup_array_index,
        lookup_array_type, lookup_array, lookup_array_raw, match_type=1
):
    res = [Error.errors['#N/A']]

    b = lookup_value_type == lookup_array_type
    index = lookup_array_index[b]
    array = lookup_array[b]

    if match_type > 0:
        def check(j, x, val, r):
            if x <= val:
                r[0] = j
                return x == val and j > 1
            return j > 1

    elif match_type < 0:
        def check(j, x, val, r):
            if x < val:
                return True
            r[0] = j
            return v == val

    else:
        if lookup_value_type == 1 and any(v in lookup_value for v in '*~?'):
            array = lookup_array_raw[b]
            lookup_value = lookup_value_raw

            def sub(m):
                return {'\\': '', '?': '.', '*': '.*'}[m.groups()[0]]

            match = regex.compile(r'^%s$' % regex.sub(
                r'(?<!\\\~)\\(?P<sub>[\*\?])|(?P<sub>\\)\~(?=\\[\*\?])',
                sub,
                regex.escape(lookup_value)
            ), regex.IGNORECASE).match

            # noinspection PyUnusedLocal
            def check(j, x, val, r):
                if match(x):
                    r[0] = j
                    return True
        else:
            b = lookup_value == array
            if b.any():
                return index[b][0]
            return Error.errors['#N/A']

    for i, v in zip(index, array):
        if check(i, v, lookup_value, res):
            break
    return res[0]


def xxmatch(
        lookup_value_type, lookup_value, lookup_value_raw, lookup_array_index,
        lookup_array_type, lookup_array, lookup_array_raw, match_type=1,
        search_mode=1
):
    if lookup_value is None:
        lookup_value_type = 3
        lookup_value = lookup_value_raw = -1
    match = None
    index = lookup_array_index
    array = lookup_array
    array_type = lookup_array_type
    value = lookup_value
    if match_type in (0, 2):
        if lookup_value_type == 1 and match_type == 2:
            array = lookup_array_raw
            value = lookup_value_raw

            def sub(m):
                return {'\\': '', '?': '.', '*': '.*'}[m.groups()[0]]

            match = regex.compile(r'^%s$' % regex.sub(
                r'(?<!\\\~)\\(?P<sub>[\*\?])|(?P<sub>\\)\~(?=\\[\*\?])',
                sub,
                regex.escape(value)
            ), regex.IGNORECASE).match
        else:
            match_type = 0
        if abs(search_mode) == 1:
            b = lookup_value_type == lookup_array_type
            array_type = lookup_array_type[b]
            array = array[b]
            index = index[b]

    array = list(zip(array_type, array))
    value = lookup_value_type, value

    best = [Error.errors['#N/A'], None]
    if match_type == -1:
        def stop(i, x, force_update=False):
            if x == value:
                best[0], best[1] = i, x
                return True
            elif x < value:
                if force_update or best[1] is None or best[1] < x:
                    best[0], best[1] = i, x
            return False
    elif match_type == 1:
        def stop(i, x, force_update=False):
            if x == value:
                best[0], best[1] = i, x
                return True
            elif x > value:
                if force_update or best[1] is None or best[1] > x:
                    best[0], best[1] = i, x
            return False
    elif match_type == 2:
        def stop(i, x, force_update=False):
            if x[0] == lookup_value_type and match(x[1]):
                best[0], best[1] = i, x
                return True
            return False
    else:
        def stop(i, x, force_update=False):
            if x == value:
                best[0], best[1] = i, x
                return True
            return False
    if search_mode == 1:
        for i, v in zip(index, array):
            if stop(i, v):
                break
    elif search_mode == -1:
        for i, v in reversed(list(zip(index, array))):
            if stop(i, v):
                break
    elif search_mode == 2:
        _binary_search(index, array, value, stop, asc=True)
    elif search_mode == -2:
        _binary_search(index, array, value, stop, asc=False)
    return best[0]


_vect_get_type_id = np.vectorize(_get_type_id, otypes=[int])

_casefold = np.vectorize(str.casefold)
_lower = np.vectorize(str.lower)


def args_parser_match_array(val, arr, match_type=1, lower=_lower):
    val_raw = np.asarray(replace_empty(val), dtype=object)
    val = val_raw.copy()
    val_types = _vect_get_type_id(val)
    b = val_types == 1
    if b.any():
        val[b] = lower(val[b].astype(str))
    lookup_array_raw = np.ravel(arr)
    lookup_array = lookup_array_raw.copy()
    arr_types = _vect_get_type_id(lookup_array)
    b = arr_types == 1
    if b.any():
        lookup_array[b] = lower(lookup_array[b].astype(str))
    index = np.arange(1, lookup_array.size + 1)
    return (
        val_types, val, val_raw, index, arr_types, lookup_array,
        lookup_array_raw, next(flatten([match_type], None))
    )


def args_parser_xmatch_array(val, arr, match_mode=0, search_mode=1):
    (
        val_types, val, val_raw, index, arr_types, lookup_array,
        lookup_array_raw
    ) = args_parser_match_array(
        replace_empty(val, None), arr, lower=_casefold
    )[:-1]

    return (
        val_types, val, val_raw, index, arr_types, lookup_array,
        lookup_array_raw, replace_empty(match_mode),
        replace_empty(search_mode, 1)
    )


def input_parser_xmatch(
        val_types, val, val_raw, index, arr_types, lookup_array,
        lookup_array_raw, match_mode, search_mode):
    match_mode = math.floor(match_mode)
    search_mode = math.floor(search_mode)
    if not (-2 < match_mode <= 3) or (
            abs(match_mode) == 2 and abs(search_mode) == 2
    ) or search_mode not in (1, 2, -1, -2):
        raise FoundError(err=Error.errors['#VALUE!'])
    if match_mode == 3:
        match_mode = 2

    return (
        val_types, val, val_raw, index, arr_types, lookup_array,
        lookup_array_raw, match_mode, search_mode
    )


FUNCTIONS['MATCH'] = wrap_ufunc(
    xmatch, check_error=lambda *a: get_error(a[1]), excluded={3, 4, 5, 6, 7},
    args_parser=args_parser_match_array, input_parser=lambda *a: a
)

FUNCTIONS['_XLFN.XMATCH'] = FUNCTIONS['XMATCH'] = wrap_ufunc(
    xxmatch,
    check_error=lambda *a: get_error(a[1]), excluded={3, 4, 5, 6},
    args_parser=args_parser_xmatch_array, input_parser=input_parser_xmatch
)


def xfilter(array, condition, if_empty=Error.errors['#VALUE!']):
    raise_errors(condition)
    array = np.asarray(array, object)
    b = np.asarray(condition, object, copy=True)
    a_shp = array.shape
    c_shp = b.shape or (1,)
    if not ((len(c_shp) == 1 or (
            len(c_shp) == 2 and 1 in c_shp
    )) and 1 <= len(a_shp) <= 2):
        return Error.errors['#VALUE!']
    b = b.ravel()
    str_type = _vect_get_type_id(b) == 1
    is_empty = np.array(sh.EMPTY, dtype=object) == b
    str_type[is_empty] = False
    b[is_empty] = False

    if str_type.any():
        return Error.errors['#VALUE!']

    b = b.astype(bool)

    for i in (0, 1):
        j = 1 - i
        if len(c_shp) == 1:
            if c_shp[0] != a_shp[i]:
                continue
        elif not (c_shp[i] == a_shp[i] and c_shp[j] == 1):
            continue
        res = array[b, :] if i == 0 else array[:, b]
        break
    else:
        return Error.errors['#VALUE!']

    if res.size == 0:
        return if_empty

    return res.view(Array)


FUNCTIONS['_XLFN._XLWS.FILTER'] = FUNCTIONS['FILTER'] = wrap_func(xfilter)


def args_parser_lookup_array(
        lookup_val, lookup_vec, result_vec=None, match_type=1):
    result_vec = np.ravel(lookup_vec if result_vec is None else result_vec)
    return args_parser_match_array(
        lookup_val, lookup_vec, match_type
    ) + (result_vec,)


def xlookup(
        lookup_value_type, lookup_value, lookup_value_raw, lookup_array_index,
        lookup_array_type, lookup_array, lookup_array_raw, match_type=1,
        result_vec=None
):
    r = xmatch(
        lookup_value_type, lookup_value, lookup_value_raw, lookup_array_index,
        lookup_array_type, lookup_array, lookup_array_raw, match_type
    )
    if not isinstance(r, XlError):
        r = np.asarray(result_vec[r - 1], object).ravel()[0]
    return r


FUNCTIONS['LOOKUP'] = wrap_ufunc(
    xlookup,
    input_parser=lambda *a: a,
    args_parser=args_parser_lookup_array,
    check_error=lambda *a: get_error(a[1]), excluded={3, 4, 5, 6, 7, 8}
)


def args_parser_xlookup_array(
        lookup_val, lookup_vec, return_array, if_not_found=Error.errors['#N/A'],
        match_mode=0, search_mode=1):
    return args_parser_xmatch_array(
        lookup_val, lookup_vec, match_mode, search_mode
    ) + (if_not_found, np.ravel(return_array))


def xxlookup(
        lookup_value_type, lookup_value, lookup_value_raw, lookup_array_index,
        lookup_array_type, lookup_array, lookup_array_raw, match_mode=0,
        search_mode=1, if_not_found=Error.errors['#N/A'], return_array=None,
):
    r = xxmatch(
        lookup_value_type, lookup_value, lookup_value_raw, lookup_array_index,
        lookup_array_type, lookup_array, lookup_array_raw, match_mode,
        search_mode
    )
    if isinstance(r, XlError):
        return if_not_found
    else:
        r = np.asarray(return_array[r - 1], object).ravel()[0]

    return r


def input_parser_xlookup(
        val_types, val, val_raw, index, arr_types, lookup_array,
        lookup_array_raw, match_mode, search_mode, if_not_found, return_array):
    return input_parser_xmatch(
        val_types, val, val_raw, index, arr_types, lookup_array,
        lookup_array_raw, match_mode, search_mode
    ) + (if_not_found, return_array)


FUNCTIONS['_XLFN.XLOOKUP'] = FUNCTIONS['XLOOKUP'] = wrap_ufunc(
    xxlookup,
    check_error=lambda *a: get_error(a[1]), excluded={3, 4, 5, 6, 10},
    args_parser=args_parser_xlookup_array, input_parser=input_parser_xlookup
)


def args_parser_hlookup(val, vec, index, match_type=1, transpose=False):
    index = int(_text2num(np.ravel(index)[0]) - 1)
    vec = np.atleast_2d(vec)
    if transpose:
        vec = vec.T
    try:
        ref = vec[index].ravel()
    except IndexError:
        raise FoundError(err=Error.errors['#REF!'])
    vec = vec[0].ravel()
    return args_parser_lookup_array(val, vec, ref, bool(match_type))


FUNCTIONS['HLOOKUP'] = wrap_ufunc(
    xlookup, input_parser=lambda *a: a,
    args_parser=args_parser_hlookup,
    check_error=lambda *a: get_error(a[1]), excluded={3, 4, 5, 6, 7, 8}
)
FUNCTIONS['VLOOKUP'] = wrap_ufunc(
    xlookup, input_parser=lambda *a: a,
    args_parser=functools.partial(args_parser_hlookup, transpose=True),
    check_error=lambda *a: get_error(a[1]), excluded={3, 4, 5, 6, 7, 8}
)


def xtranspose(array):
    return np.transpose(array).view(Array)


FUNCTIONS['TRANSPOSE'] = wrap_func(xtranspose)
