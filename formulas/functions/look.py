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
    elif isinstance(obj, XlError):
        return next((
            i for i, k in enumerate(Error.errors.values()) if k == obj
        ), -1) + 4
    elif obj is sh.EMPTY:
        return 4 + len(Error.errors)
    elif isinstance(obj, (str, np.str_)):
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


def xunique(array, by_col=0, exactly_once=0):
    raise_errors(by_col, exactly_once)
    by_col = int(_convert2float(by_col))
    exactly_once = int(exactly_once)
    array = np.atleast_2d(array)

    if by_col:
        array = array.T

    array = [tuple((isinstance(x, bool), x) for x in row) for row in array]
    if exactly_once:
        counts = collections.Counter(array)
        array = [v for v in array if counts[v] == 1]
    else:
        unique = set()
        array = [v for v in array if v not in unique and (unique.add(v) or 1)]
    if not array:
        return None
    array = [tuple(x[1] for x in row) for row in array]
    if by_col:
        array = np.array(array, object).T.tolist()
    return array


def return_unique_func(res, *args):
    if not res.shape:
        res = res.item()
        res = np.asarray(res, object).view(Array)
    elif res.shape == (1, 1):
        res = res[0, 0]
        res = np.asarray(res, object).view(Array)
    else:
        v = res[0, 0]
        res = np.vectorize(lambda x: get_error(x) or 0, otypes=[object])(res)
        res[0, 0] = v[0][0] if isinstance(v, list) else v
    res[res == None] = Error.errors['#VALUE!']
    return res.view(Array)


FUNCTIONS['_XLFN.UNIQUE'] = FUNCTIONS['UNIQUE'] = wrap_ufunc(
    xunique,
    excluded={0},
    check_nan=False,
    check_error=lambda array, *a: None,
    input_parser=lambda *a: a,
    args_parser=lambda array, by_col=0, exactly_once=0: (
        array, replace_empty(by_col), replace_empty(exactly_once)
    ),
    return_func=return_unique_func
)


def args_parser_typed_array(arr):
    arr = np.atleast_2d(arr)
    val_types = _vect_get_type_id(arr)
    return np.stack([val_types, arr], axis=-1)


def xsort(array, sort_index=1, sort_order=1, by_col=0):
    raise_errors(sort_index, sort_order, by_col)
    array = args_parser_typed_array(array)
    sort_index = int(replace_empty(sort_index))
    sort_order = int(replace_empty(sort_order))
    by_col = int(_convert2float(replace_empty(by_col)))

    if by_col:
        array = np.swapaxes(array, 0, 1)

    if sort_order in (1, -1) and 1 <= sort_index <= array.shape[1]:
        sort_index -= 1
        array = sorted(
            [tuple(tuple(v) for v in row) for row in array],
            key=lambda x: x[sort_index], reverse=sort_order != 1,
        )
    else:
        return Error.errors['#VALUE!']

    array = np.array([tuple(x[1] for x in row) for row in array], object)
    if by_col:
        array = array.T
    return array.view(Array)


FUNCTIONS['_XLFN._XLWS.SORT'] = FUNCTIONS['SORT'] = wrap_func(xsort)


def xsortby(ref, by_array, sort_order1=1, *args):
    ref = np.atleast_2d(ref)
    raise_errors(sort_order1, args[1::2])
    shape = by_array.shape
    by_col = int(shape[0] == 1)
    if 1 not in shape or any(
            shape != a.shape for a in args[::2]
    ) or ref.shape[by_col] != shape[by_col]:
        return Error.errors['#VALUE!']

    args = (by_array, sort_order1) + args
    n = max(shape)
    indices = []
    for a, sort_order in zip(args[::2], args[1::2]):
        sort_order = int(replace_empty(sort_order))
        if sort_order not in (1, -1):
            return Error.errors['#VALUE!']
        unique = {}
        it = enumerate(map(tuple, args_parser_typed_array(a.ravel())[0]))
        for i, v in it:
            if v in unique:
                unique[v].append(i)
            else:
                unique[v] = [i]
        index = np.empty(n, dtype=int)
        for i, v in enumerate(sorted(unique, reverse=sort_order != 1)):
            index[unique[v]] = i
        indices.append(index)

    index = np.arange(n, dtype=int)
    indices.append(index)
    indices = np.column_stack(indices)
    index = sorted(index, key=lambda x: tuple(indices[x]))
    return (ref[:, index] if by_col else ref[index]).view(Array)


FUNCTIONS['_XLFN.SORTBY'] = FUNCTIONS['SORTBY'] = wrap_func(xsortby)


def to_python(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        if any(v != 1 for v in obj.shape):
            return Error.errors['#VALUE!']
        return obj.ravel()[0]
    return obj


def _last_if_tuple(x):
    return x[-1] if isinstance(x, tuple) else x


def map_multiindex_take_last_if_tuple(mi):
    """
    For each label in a (Multi)Index, if that label is a tuple, replace it
    with its second element. Works for both Index and MultiIndex.
    """
    import pandas as pd
    if isinstance(mi, pd.MultiIndex):
        new_tuples = [
            tuple(_last_if_tuple(v) for v in tup)
            for tup in mi.to_list()
        ]
        return pd.MultiIndex.from_tuples(new_tuples, names=mi.names)
    else:
        return mi.map(_last_if_tuple)


def _xpivotby(
        row_fields, col_fields, values, aggfunc, field_headers=None,
        row_total_depth=None, row_sort_order=1, col_total_depth=None,
        col_sort_order=1, filter_array=None, relative_to=0, _groupby=False):
    raise_errors(
        aggfunc, field_headers, row_total_depth, row_sort_order,
        col_total_depth, col_sort_order, relative_to
    )
    values = np.atleast_2d(values)
    row_fields = np.atleast_2d(row_fields)
    col_fields = np.atleast_2d(col_fields)
    if not (values.shape[0] == row_fields.shape[0] == col_fields.shape[0]):
        return Error.errors['#VALUE!']

    if field_headers is None:
        if _get_type_id(values.ravel()[0]) == 1:
            field_headers = 1
        else:
            field_headers = 0

        if (
                row_fields.shape[1] > 1 or values.shape[1] > 1
                or col_fields.shape[1] > 1
        ) and not _groupby:
            field_headers += 2

    if row_total_depth is None:
        row_total_depth = row_fields.shape[1]
    if col_total_depth is None:
        col_total_depth = col_fields.shape[1]

    add_totalset = False
    try:
        raise_errors(to_python(aggfunc([1, 2])))
        agg = lambda s, *args: '' if (s == '').all() else to_python(
            aggfunc(s)
        )
    except Exception:
        if get_error(to_python(aggfunc([1, 1], [1, 1]))):
            agg = lambda s, *args: '' if (s == '').all() else Error.errors[
                '#VALUE!'
            ]
        else:
            agg = lambda s, *args: '' if (s == '').all() else to_python(
                aggfunc(s, *args)
            )
            add_totalset = True

    import pandas as pd
    parse = args_parser_typed_array
    data = {}
    pivot = {}
    has_headers = field_headers in (1, 3)

    for k, it in {
        'Row Field': row_fields,
        'Column Field': col_fields,
        'Value': values
    }.items():
        for i, v in enumerate(it.T, 1):
            i = f'{k} {i}'
            if k != 'Value':
                v = list(map(tuple, parse(v)[0]))
                j = v[0][1] if has_headers else i
            else:
                j = v[0] if has_headers else i
            sh.get_nested_dicts(
                pivot, k, default=collections.OrderedDict
            )[i] = j
            data[i] = v[int(has_headers):]

    data = pd.DataFrame(data)
    if filter_array is not None:
        filter_array = replace_empty(filter_array)
        filter_array = np.atleast_2d(filter_array)[int(has_headers):, 0]
        b = _vect_get_type_id(filter_array)
        if not np.isin(b, (0, 2)).all():
            raise FoundError(err=Error.errors['#VALUE!'])
        data = data.iloc[filter_array.astype(bool).ravel(), :]
    df = []
    rows = list(pivot['Row Field'])
    cols = list(pivot['Column Field'])
    values = list(pivot['Value'])
    totalset = None

    col_sort_order = np.asarray(col_sort_order, dtype=int).ravel()
    abs_col = set(np.abs(col_sort_order))
    sort_col_by_value = False
    remove_row_total_depth = False
    if col_sort_order.size == 1 and abs(col_sort_order.item()) == len(cols) + 1:
        sort_col_by_value = True
        col_sort_order = {cols[-1]} if col_sort_order.item() < 0 else set()
        if not row_total_depth:
            row_total_depth = 1
            remove_row_total_depth = True
    elif len(abs_col) != col_sort_order.size or abs_col - set(
            range(1, len(cols) + 1)
    ):
        raise FoundError(err=Error.errors['#VALUE!'])
    else:
        col_sort_order = {cols[i] for i in (
                -col_sort_order[col_sort_order < 0] - 1
        )}

    row_sort_order = np.asarray(row_sort_order).ravel()
    abs_row = set(np.abs(row_sort_order))
    sort_row_by_value = False
    remove_col_total_depth = False
    if row_sort_order.size == 1 and abs(row_sort_order.item()) == len(rows) + 1:
        sort_row_by_value = True
        row_sort_order = {rows[-1]} if row_sort_order.item() < 0 else set()
        if not col_total_depth:
            col_total_depth = 1
            remove_col_total_depth = True
    elif len(abs_row) != row_sort_order.size or abs_row - set(
            range(1, len(rows) + 1)
    ):
        raise FoundError(err=Error.errors['#VALUE!'])
    else:
        row_sort_order = {rows[i] for i in (
                -row_sort_order[row_sort_order < 0] - 1
        ).astype(int)}
    for value in values:
        if add_totalset:
            if relative_to in (3, 4):
                _totalset = {}
                for i in range(0, len(cols) + 1):
                    for j in range(0, len(rows) + 1):
                        keys = cols[:i] + rows[:j]
                        dat = data[[value] + keys]
                        if keys:
                            for k, v in dat.groupby(keys).groups.items():
                                k = (k,) if len(keys) == 1 else k
                                _totalset[(k[:i], k[i:])] = dat.loc[
                                    v, value
                                ].values
                        else:
                            _totalset[((), ())] = dat.values
            elif relative_to == 2:  # 2: Grand Totals
                totalset = data[[value]].values
            else:
                _totalset = {(): data[[value]].values}
        it_c = sorted(set(
            list(range(0, int(abs(col_total_depth)))) + [len(cols)]))
        it_r = sorted(set(
            list(range(0, int(abs(row_total_depth)))) + [len(rows)]))
        for i in it_c:
            if add_totalset and relative_to == 0 and i:  # 0: Column Totals(Default)
                _totalset = {
                    k: v[[value]].values for k, v in
                    data[[value] + cols[:i]].groupby(cols[:i])
                }
                _totalset[()] = data[[value]].values
            for j in it_r:
                if add_totalset and relative_to == 1 and j:  # 1: Row Totals
                    _totalset = {
                        k: v[[value]].values
                        for k, v in data[[value] + rows[:j]].groupby(
                            rows[:j]
                        )
                    }
                    _totalset[()] = data[[value]].values

                keys = cols[:i] + rows[:j]
                dat = data[[value] + keys]

                if keys:
                    it = ((
                        (k,) if len(keys) == 1 else k,
                        dat.loc[v, value].values
                    ) for k, v in dat.groupby(keys).groups.items())
                else:
                    it = [((), dat.values)]
                for k, v in it:
                    if add_totalset:
                        if relative_to == 0:
                            totalset = _totalset[k[:i]]
                        elif relative_to == 1:
                            totalset = _totalset[k[i:]]
                        elif relative_to == 3:  # 3: Parent Col Total
                            totalset = _totalset[(k[:max(i - 1, 0)], k[i:])]
                        elif relative_to == 4:  # 4: Parent Row Total
                            totalset = _totalset[
                                (k[:i], (k[i:-1] if j > 1 else ()))
                            ]

                    v = agg(v, totalset)
                    df.append({
                        'Value': value, 'agg': v, **sh.map_list(keys, *k)
                    })

    df = pd.DataFrame(df)

    tot_row_key = (
        -50 if row_total_depth < 0 else 50,
        f'{"Grand " if abs(row_total_depth) > 1 else ""}Total'
    )
    tot_col_key = (
        -50 if col_total_depth < 0 else 50,
        f'{"Grand " if abs(col_total_depth) > 1 else ""}Total'
    )

    for i in rows:
        neg = (-1 if i in row_sort_order else 1)
        tot_key = neg * tot_row_key[0], tot_row_key[1]
        df[i] = df[i].map(lambda x: tot_key if pd.isna(x) else x)
        tot_row_key = tot_row_key[0], ''
    for i in cols:
        neg = (-1 if i in col_sort_order else 1)
        tot_key = neg * tot_col_key[0], tot_col_key[1]
        df[i] = df[i].map(lambda x: tot_col_key if pd.isna(x) else x)
        tot_col_key = tot_col_key[0], ''

    with pd.option_context("future.no_silent_downcasting", True):
        df = pd.pivot_table(
            df, index=rows, columns=cols + ['Value'], values='agg',
            aggfunc=lambda x: x, fill_value='', sort=False
        )
    if sort_row_by_value:
        b = np.asarray([
            abs(x[0]) == 50 for x in df.columns.get_level_values(cols[0])
        ])
        df['sort'] = [
            (i[0] if abs(i[0]) == 50 else 0, v, i) for v, i in zip(
                df.iloc[:, b].values.ravel(),
                df.index.get_level_values(rows[-1])
            )
        ]
        df.sort_values(by=rows[:-1] + ['sort'], ascending=[
            k not in row_sort_order for k in rows
        ], axis=0, inplace=True)
        df.drop('sort', axis=1, inplace=True)
        if remove_col_total_depth:
            df = df.iloc[:, ~b]
    else:
        df.sort_values(axis=0, by=rows, ascending=[
            k not in row_sort_order for k in rows
        ], inplace=True)
    if sort_col_by_value:
        b = np.asarray([
            abs(x[0]) == 50 for x in df.index.get_level_values(rows[0])
        ])
        v = [(i[0] if abs(i[0]) == 50 else 0, v, i) for v, i in zip(
            df.iloc[b].values.ravel(),
            df.columns.get_level_values(cols[-1])
        )]
        df = df.T
        df['sort'] = v
        df = df.T
        df.sort_values(by=cols[:-1] + ['sort', 'Value'], ascending=[
            k not in col_sort_order for k in cols + ['Value']
        ], axis=1, inplace=True)
        df.drop('sort', axis=0, inplace=True)
        if remove_row_total_depth:
            df = df.iloc[~b, :]

    else:
        df.sort_values(axis=1, by=cols + ['Value'], ascending=[
            k not in col_sort_order for k in (cols + ['Value'])
        ], inplace=True)

    df.columns = df.columns.set_levels(
        df.columns.levels[-1].map(pivot['Value']), level=-1
    )
    df.columns = map_multiindex_take_last_if_tuple(df.columns)
    df.index = map_multiindex_take_last_if_tuple(df.index)

    columns_names = [
        pivot['Column Field'].get(k, k) for k in df.columns.names
    ]
    columns = df.columns.to_frame().iloc[:, :-1].values.T
    index_names = [pivot['Row Field'].get(k, k) for k in df.index.names]
    n = len(index_names)
    index_names.extend(df.columns.get_level_values(-1))

    df.fillna(value='', inplace=True)
    df.reset_index(inplace=True)
    header = np.empty((len(columns_names) + 1, df.shape[1]), dtype=object)
    header[:] = ''
    header[1:-1, n:] = columns
    if field_headers >= 2:
        header[0, n] = ', '.join(map(str, columns_names[:-1]))
        header[-1] = index_names
        if _groupby:
            header = header[2:]
    elif _groupby:
        header = header[3:-1]
    else:
        header = header[1:-1]
    return np.vstack((header, df.map(to_python).values)).view(Array)


FUNCTIONS['_XLFN.PIVOTBY'] = FUNCTIONS['PIVOTBY'] = wrap_func(functools.partial(
    _xpivotby, _groupby=False
))


def xgroupby(
        row_fields, values, aggfunc, field_headers=None, row_total_depth=1,
        row_sort_order=1, filter_array=None, field_relationship=0
):
    values = np.atleast_2d(values)
    col_fields = np.ones((values.shape[0], 1), dtype=int)
    return _xpivotby(
        row_fields, col_fields, values, aggfunc, field_headers=field_headers,
        row_total_depth=row_total_depth, row_sort_order=row_sort_order,
        filter_array=filter_array, col_total_depth=0, _groupby=True
    )


FUNCTIONS['_XLFN.GROUPBY'] = FUNCTIONS['GROUPBY'] = wrap_func(xgroupby)


def xdrop(array, rows, columns=0):
    columns = int(_convert2float(columns or 0))
    if array.shape[0] < abs(rows) or array.shape[1] < abs(columns):
        raise FoundError(err=Error.errors['#VALUE!'])
    r = slice(rows, None) if rows >= 0 else slice(rows)
    c = slice(columns, None) if columns >= 0 else slice(columns)
    return array[r, c].tolist()


def xtake(array, rows, columns=None):
    if columns is None:
        columns = array.shape[1]
    else:
        columns = int(_convert2float(columns))
    if columns == 0 or rows == 0:
        raise FoundError(err=Error.errors['#VALUE!'])
    r = slice(rows) if rows >= 0 else slice(rows, None)
    c = slice(columns) if columns >= 0 else slice(columns, None)
    return array[r, c].tolist()


kw_drop = dict(
    input_parser=lambda arr, rows, columns: (
        arr, int(_convert2float(rows)), columns
    ),
    check_error=lambda arr, rows, columns=None: get_error(rows, columns),
    args_parser=lambda arr, rows, columns=None: (
        np.atleast_2d(arr), replace_empty(rows), replace_empty(columns)
    ), return_func=return_2d_func, check_nan=False, excluded={0}
)
FUNCTIONS['_XLFN.DROP'] = FUNCTIONS['DROP'] = wrap_ufunc(xdrop, **kw_drop)
FUNCTIONS['_XLFN.TAKE'] = FUNCTIONS['TAKE'] = wrap_ufunc(xtake, **kw_drop)


def xtrimrange(array, trim_rows, trim_cols):
    b = array == np.array(sh.EMPTY, dtype=object)
    p = {0: {'i': 0}, 1: {'i': 0}}
    p[0]['j'], p[1]['j'] = array.shape
    for axis, trim in ((0, trim_rows), (1, trim_cols)):
        if trim == 0:
            pass
        elif trim in (1, 2, 3):
            n = np.arange(array.shape[axis])[~b.all(axis=1 if axis == 0 else 0)]
            if not n.size:
                raise FoundError(err=Error.errors['#REF!'])
            if trim in (1, 3):
                p[axis]['i'] = np.min(n)
            if trim_rows in (2, 3):
                p[axis]['j'] = np.max(n) + 1
        else:
            raise FoundError(err=Error.errors['#VALUE!'])
    return array[p[0]['i']:p[0]['j'], p[1]['i']:p[1]['j']].tolist()


def return_trimrange_func(res, *args):
    if not res.shape or res.shape == (1, 1):
        return return_2d_func(res, *args)
    return Error.errors['#VALUE!']


FUNCTIONS['_XLFN.TRIMRANGE'] = FUNCTIONS['TRIMRANGE'] = wrap_ufunc(
    xtrimrange, input_parser=lambda arr, trim_rows, trim_cols: (
        arr, _convert2float(trim_rows), _convert2float(trim_cols)
    ),
    check_error=lambda arr, trim_rows=0, trim_cols=0: get_error(
        trim_rows, trim_cols
    ),
    args_parser=lambda arr, trim_rows=0, trim_cols=0: (
        np.atleast_2d(arr), replace_empty(trim_rows), replace_empty(trim_cols)
    ), return_func=return_trimrange_func, check_nan=False, excluded={0}
)


def xexpand(array, rows, columns=None, pad_with=Error.errors['#N/A']):
    r, c = array.shape
    columns = c if columns is None else int(_convert2float(columns))
    if r > rows or c > columns:
        raise FoundError(err=Error.errors['#VALUE!'])
    res = np.empty((rows, columns), object)
    res[:, :] = pad_with
    res[:r, :c] = array
    return res.tolist()


FUNCTIONS['_XLFN.EXPAND'] = FUNCTIONS['EXPAND'] = wrap_ufunc(
    xexpand, input_parser=lambda arr, rows, columns, pad_with: (
        arr, int(_convert2float(rows)), columns, pad_with
    ),
    check_error=lambda arr, rows, columns, pad_with: get_error(rows, columns),
    args_parser=lambda arr, rows, columns=None, pad_with=Error.errors['#N/A']: (
        np.atleast_2d(arr), replace_empty(rows), replace_empty(columns),
        pad_with
    ), return_func=return_2d_func, check_nan=False, excluded={0}
)


def xhstack(array, *arrays):
    arrays = tuple(map(np.atleast_2d, (array,) + arrays))
    tc = 0
    tr = 0
    for arr in arrays:
        r, c = arr.shape
        tc += c
        tr = max(r, tr)
    res = np.empty((tr, tc), object)
    res[:, :] = Error.errors['#N/A']
    tc = 0
    for arr in arrays:
        r, c = arr.shape
        res[:r, tc:tc + c] = arr
        tc += c
    return res.view(Array)


def xvstack(array, *arrays):
    return xhstack(*(np.atleast_2d(a).T for a in ((array,) + arrays))).T


FUNCTIONS['_XLFN.HSTACK'] = FUNCTIONS['HSTACK'] = wrap_func(xhstack)
FUNCTIONS['_XLFN.VSTACK'] = FUNCTIONS['VSTACK'] = wrap_func(xvstack)


def xwrapcols(array, wrap_count, pad_with, cols=True):
    if wrap_count <= 0:
        raise FoundError(err=Error.errors['#NUM!'])
    i = int(np.ceil(array.size / wrap_count))
    need = i * wrap_count - array.size
    if need:
        array = np.pad(
            array, (0, need), mode='constant', constant_values=pad_with
        )
    array = array.reshape((i, wrap_count))
    if cols:
        array = array.T
    return array.tolist()


FUNCTIONS['_XLFN.WRAPROWS'] = FUNCTIONS['WRAPROWS'] = wrap_ufunc(
    functools.partial(xwrapcols, cols=False),
    input_parser=lambda arr, wrap_count, pad_with: (
        arr, int(_convert2float(wrap_count)), pad_with
    ),
    check_error=lambda arr, wrap_count, pad_with: get_error(wrap_count),
    args_parser=lambda arr, wrap_count, pad_with=Error.errors['#N/A']: (
        np.asarray(arr, object).ravel(), replace_empty(wrap_count), pad_with
    ), return_func=return_2d_func, check_nan=False, excluded={0}
)
FUNCTIONS['_XLFN.WRAPCOLS'] = FUNCTIONS['WRAPCOLS'] = wrap_ufunc(
    xwrapcols, input_parser=lambda arr, wrap_count, pad_with: (
        arr, int(_convert2float(wrap_count)), pad_with
    ),
    check_error=lambda arr, wrap_count, pad_with: get_error(wrap_count),
    args_parser=lambda arr, wrap_count, pad_with=Error.errors['#N/A']: (
        np.asarray(arr, object).ravel(), replace_empty(wrap_count), pad_with
    ), return_func=return_2d_func, check_nan=False, excluded={0}
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
