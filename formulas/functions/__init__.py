#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2025 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
It provides functions implementations to compile the Excel functions.

Sub-Modules:

.. currentmodule:: formulas.functions

.. autosummary::
    :nosignatures:
    :toctree: functions/

    ~comp
    ~date
    ~eng
    ~financial
    ~google
    ~info
    ~logic
    ~look
    ~math
    ~operators
    ~stat
    ~text
"""
import re
import copy
import importlib
import functools
import collections
import numpy as np
import schedula as sh
from collections.abc import Iterable
from formulas.errors import (
    RangeValueError, FoundError, BaseError, BroadcastError, InvalidRangeError
)
from formulas.tokens.operand import Error, XlError

COMPILING = sh.Token('Run')


def get_shape(r=1, c=1):
    r = None if r == 1 else r
    c = None if c == 1 else c
    return r, c


def _init_reshape(base_shape, value):
    res = np.empty(base_shape, object)
    res[:, :] = getattr(value, '_default', Error.errors['#N/A'])
    r, c = get_shape(*value.shape)
    return res, r, c


class Array(np.ndarray):
    _default = Error.errors['#N/A']

    _collapse_value = None

    def reshape(self, shape, *shapes, order='C'):
        try:
            # noinspection PyArgumentList
            return super(Array, self).reshape(shape, *shapes, order=order)
        except ValueError:
            res, r, c = _init_reshape(shape, self)
            try:
                res[:r, :c] = self
            except ValueError:
                res[:, :] = self.collapse(shape)
            return res

    def collapse(self, shape):
        if self._collapse_value is not None and \
                tuple(shape) == (1, 1) != self.shape:
            return self._collapse_value
        return np.resize(self, shape)

    def __reduce__(self):
        reduce = super(Array, self).__reduce__()  # Get the parent's __reduce__.
        state = {
            '_collapse_value': self._collapse_value,
            '_default': self._default
        },  # Additional state params to pass to __setstate__.
        return reduce[0], reduce[1], reduce[2] + state

    def __setstate__(self, state, *args, **kwargs):
        self.__dict__.update(state[-1])  # Set the attributes.
        super(Array, self).__setstate__(state[0:-1], *args, **kwargs)

    def __deepcopy__(self, memo=None, *args, **kwargs):
        obj = super(Array, self).__deepcopy__(memo, *args, **kwargs)
        # noinspection PyArgumentList
        obj._collapse_value = copy.deepcopy(self._collapse_value, memo)
        # noinspection PyArgumentList
        obj._default = copy.deepcopy(self._default, memo)
        return obj

    def __hash__(self):
        return hash(self.tolist())


# noinspection PyUnusedLocal
def not_implemented(*args, **kwargs):
    raise NotImplementedError


def replace_empty(x, empty=0):
    if isinstance(x, np.ndarray):
        obj = np.array(sh.EMPTY, dtype=object)
        if obj in x:
            x = np.where(obj == x, empty, x).view(x.__class__)
    elif x is sh.EMPTY:
        return empty
    return x


def is_not_empty(v):
    return v is not sh.EMPTY


def wrap_impure_func(func):
    def wrapper(compiling, *args, **kwargs):
        return sh.NONE if compiling else func(*args, **kwargs)

    return functools.update_wrapper(wrapper, func)


# noinspection PyUnusedLocal
def wrap_func(func, ranges=False):
    def wrapper(*args, **kwargs):
        # noinspection PyBroadException
        try:
            return func(*args, **kwargs)
        except FoundError as ex:
            return np.asarray([[ex.err]], object)
        except InvalidRangeError:
            return np.asarray([[Error.errors['#VALUE!']]], object)
        except BaseError as ex:
            raise ex
        except Exception:
            return np.asarray([[Error.errors['#VALUE!']]], object)

    if not ranges:
        return wrap_ranges_func(functools.update_wrapper(wrapper, func))
    return functools.update_wrapper(wrapper, func)


def wrap_ranges_func(func, n_out=1):
    def wrapper(*args, **kwargs):
        try:
            args, kwargs = parse_ranges(*args, **kwargs)
            return func(*args, **kwargs)
        except RangeValueError:
            return sh.bypass(*((sh.NONE,) * n_out))

    return functools.update_wrapper(wrapper, func)


def parse_ranges(*args, **kw):
    from ..ranges import Ranges
    args = tuple(v.value if isinstance(v, Ranges) else v for v in args)
    kw = {k: v.value if isinstance(v, Ranges) else v for k, v in kw.items()}
    return args, kw


SUBMODULES = [
    '.info', '.logic', '.math', '.stat', '.financial', '.text', '.look', '.eng',
    '.date', '.comp', '.google'
]
# noinspection PyDictCreation
FUNCTIONS = {}
FUNCTIONS['ARRAY'] = lambda *args: np.asarray(args, object).view(Array)
FUNCTIONS['ARRAYROW'] = lambda *args: np.asarray(args, object).view(Array)


def get_error(*vals):
    # noinspection PyTypeChecker
    for v in flatten(vals, None, True):
        if isinstance(v, XlError):
            return v


def raise_errors(*args):
    # noinspection PyTypeChecker
    v = get_error(*args)
    if v:
        raise FoundError(err=v)


def _to_number(number):
    if isinstance(number, (bool, np.bool_)) and number:
        return np.nan
    try:
        return float(number)
    except (ValueError, TypeError):
        return np.nan


@functools.lru_cache(None)
def _compile_func(func):
    return np.frompyfunc(func, 1, 1)


def to_number(*args, **kwargs):
    return _compile_func(_to_number)(*args, **kwargs)


def clean_values(values):
    return values[values != np.array(sh.EMPTY, dtype=object)]


def is_number(number, xl_return=True, bool_return=False):
    if isinstance(number, (bool, np.bool_)):
        return bool_return
    elif isinstance(number, XlError):
        return xl_return
    elif number is sh.EMPTY:
        return False
    else:
        try:
            float(number)
        except (ValueError, TypeError):
            return False
    return True


def _text2num(value):
    if isinstance(value, Array) and not value.shape:
        value = value.tolist()
    if not isinstance(value, Error) and isinstance(value, str):
        try:
            return float(value)
        except (ValueError, TypeError):
            from .date import xdate, _text2datetime
            try:
                return xdate(*_text2datetime(value)[:3])
            except (FoundError, AssertionError):
                pass
    return value


def convert2float(*a):
    return map(_convert2float, a)


def _convert2float(v):
    if isinstance(v, XlError):
        raise FoundError(err=v)
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, str):
        return float(_text2num(v))
    return float(v)


def _convert_args(v):
    if isinstance(v, XlError):
        return v
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, str):
        return float(_text2num(v))
    return v


@functools.lru_cache(None)
def _text2num_vectorize():
    return np.vectorize(_text2num, otypes=[object])


def text2num(*args, **kwargs):
    return _text2num_vectorize()(*args, **kwargs)


def _get_single_args(*args):
    res = []
    for v in args:
        v = tuple(flatten(v, None))
        if len(v) != 1 or isinstance(v[0], bool):
            raise FoundError(err=Error.errors['#VALUE!'])
        res.append(v[0])
    return res


_re_condition = re.compile('(?<!~)[?*]')


def __xfilter(test_range, condition):
    from .operators import LOGIC_OPERATORS
    operator = '='
    if isinstance(condition, str):
        for k in LOGIC_OPERATORS:
            if condition.startswith(k) and condition != k:
                operator, condition = k, condition[len(k):]
                break
        if operator == '=':
            it = _re_condition.findall(condition)
            if it:
                _ = lambda v: re.escape(v.replace('~?', '?').replace('~*', '*'))
                match = re.compile(''.join(sum(zip(
                    map(_, _re_condition.split(condition)),
                    tuple(map(lambda v: '.%s' % v, it)) + ('',)
                ), ()))).match
                f = lambda v: isinstance(v, str) and bool(match(v))
                b = np.vectorize(f, otypes=[bool])(test_range['raw'])
                try:
                    return b
                except FoundError as ex:
                    return ex.err
            elif any(v in condition for v in ('~?', '~*')):
                condition = condition.replace('~?', '?').replace('~*', '*')
        from ..tokens.operand import Number, Error
        from ..errors import TokenError
        for token in (Number, Error):
            try:
                token = token(condition)
                if token.end_match == len(condition):
                    condition = token.compile()
                    break
            except TokenError:
                pass

        condition = _text2num(condition)

    from .operators import _get_type_id
    type_id = _get_type_id(condition)
    if operator == '=' and type_id ==1:
        operator = lambda x, y: x == y or x.lower() == y.lower()
    else:
        operator = LOGIC_OPERATORS[operator]

    @functools.lru_cache()
    def check(value):
        return _get_type_id(value) == type_id and operator(value, condition)

    if is_number(condition):
        if 'num' not in test_range:
            test_range['num'] = text2num(test_range['raw'])
        b = np.vectorize(check, otypes=[bool])(test_range['num'])
    else:
        b = np.vectorize(check, otypes=[bool])(test_range['raw'])
    try:
        return b
    except FoundError as ex:
        return ex.err


def _xfilter(accumulator, operating_range, test_ranges, *conditions):
    operating_range = np.asarray(operating_range)
    try:
        b = None
        for test_range, condition in zip(test_ranges, conditions):
            if b is None:
                b = __xfilter(test_range, condition)
            else:
                b &= __xfilter(test_range, condition)
        return accumulator(operating_range[b])
    except FoundError as ex:
        return ex.err


_xfilter = np.vectorize(_xfilter, otypes=[object], excluded={0, 1, 2})


def xfilter(accumulator, test_range, condition, operating_range=None):
    operating_range = test_range if operating_range is None else operating_range
    # noinspection PyTypeChecker
    test_range = {'raw': replace_empty(test_range, '')}
    res = _xfilter(accumulator, operating_range, [test_range], condition)
    return res.view(Array)


def xfilters(accumulator, operating_range, test_range, condition, *args):
    # sanity-check: the rest has to be even-length
    if len(args) % 2:
        raise ValueError(
            "Additional arguments must be supplied in (test_range, condition) pairs."
        )
    test_ranges = [test_range, *args[0::2]]
    operating_range = test_range if operating_range is None else operating_range

    # noinspection PyTypeChecker
    test_ranges = [{'raw': replace_empty(v, '')} for v in test_ranges]
    res = _xfilter(
        accumulator, operating_range, test_ranges, condition, *args[1::2]
    )
    return res.view(Array)


def flatten(v, check=is_number, drop_empty=False):
    if isinstance(v, np.ndarray):
        if drop_empty or check is is_number or check is is_not_empty:
            v = v[v != np.array(sh.EMPTY, dtype=object)]
        if not check or check is is_not_empty:
            yield from v.ravel()
        else:
            yield from filter(check, v.ravel())
    elif not isinstance(v, str) and isinstance(v, Iterable):
        for el in v:
            yield from flatten(el, check, drop_empty)
    elif not check or check(v):
        yield v


# noinspection PyUnusedLocal
def value_return(res, *args):
    res._collapse_value = Error.errors['#VALUE!']
    return res


def convert_nan(value, default=Error.errors['#NUM!']):
    return value if np.isfinite(value) else default


def convert_noshp(value):
    if isinstance(value, np.ndarray) and not value.shape:
        value = value.ravel()[0]
    return value


def args2vals(args):
    return (np.ravel(v)[0] for v in args)


def args2list(max_shape, shapes, *args):
    it = []
    for arg, shape in zip(args, shapes):
        if not shape or shape[0] == 1:
            arg = np.tile(arg, max_shape)
        elif shape[0] != max_shape:
            raise BroadcastError()
        it.append(arg)
    return map(args2vals, zip(*it))


def wrap_ufunc(
        func, input_parser=lambda *a: map(float, a), check_error=get_error,
        args_parser=lambda *a: map(replace_empty, a), otype=Array,
        ranges=False, return_func=lambda res, *args: res, check_nan=True, **kw):
    """Helps call a numpy universal function (ufunc)."""

    def safe_eval(*vals):
        try:
            r = check_error(*vals) or convert_noshp(func(*input_parser(*vals)))
            if check_nan and not isinstance(r, (XlError, str)):
                r = convert_nan(r)
        except FoundError as ex:
            r = ex.err
        except (ValueError, TypeError):
            r = Error.errors['#VALUE!']
        return r

    kw['otypes'] = kw.get('otypes', [object])

    # noinspection PyUnusedLocal
    def wrapper(*args, **kwargs):
        try:
            args = tuple(args_parser(*args))
            with np.errstate(divide='ignore', invalid='ignore'):
                if len(args) >= 32:
                    shapes = [np.shape(arg) for arg in args]
                    max_shape = max((s or (1,))[0] for s in shapes)
                    if max_shape == 1:
                        res = np.asarray([[
                            safe_eval(*args2vals(args))
                        ]], object).view(otype)
                    else:
                        res = np.asarray([safe_eval(*v) for v in args2list(
                            max_shape, shapes, *args
                        )], object).view(otype)
                else:
                    res = np.vectorize(safe_eval, **kw)(*args)
            try:
                res = res.view(otype)
            except AttributeError:
                res = np.asarray([[res]], object).view(otype)
            return return_func(res, *args)
        except ValueError as ex:
            try:
                np.broadcast(*args)
            except ValueError:
                raise BroadcastError()
            raise ex

    return wrap_func(functools.update_wrapper(wrapper, func), ranges=ranges)


@functools.lru_cache()
def get_functions():
    functions = collections.defaultdict(lambda: not_implemented)
    for name in SUBMODULES:
        functions.update(importlib.import_module(name, __name__).FUNCTIONS)
    functions.update(FUNCTIONS)
    return functions
