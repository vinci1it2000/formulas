#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2022 European Commission (JRC);
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
import typing as t
from collections.abc import Iterable
from formulas.errors import (
    RangeValueError, FoundError, BaseError, BroadcastError, InvalidRangeError
)
from formulas.tokens.operand import Error, XlError

COMPILING = sh.Token('Run')


def _init_reshape(base_shape, value):
    res, (r, c) = np.empty(base_shape, object), value.shape
    res[:, :] = getattr(value, '_default', Error.errors['#N/A'])
    r = None if r == 1 else r
    c = None if c == 1 else c
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
        y = x.ravel().tolist()
        if sh.EMPTY in y:
            y = [empty if v is sh.EMPTY else v for v in y]
            return np.asarray(y, object).reshape(*x.shape)
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
    for v in flatten(vals, None):
        if isinstance(v, XlError):
            return v


def raise_errors(*args):
    # noinspection PyTypeChecker
    v = get_error(*args)
    if v:
        raise FoundError(err=v)


def is_number(number, xl_return: bool = True) -> bool:
    """Checks if a given element is a number.

    Args:
        number (_type_): The element to be tested.
        xl_return (bool, optional): _description_. Defaults to True.

    Returns:
        bool: True if is a number, False otherwise.
    """    
    if isinstance(number, (bool, np.bool_)):
        return False
    elif isinstance(number, XlError):
        return xl_return
    else:
        try:
            float(number)
        except (ValueError, TypeError):
            return False
    return True


@functools.lru_cache()
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


def _xfilter(accumulator, test_range, condition, operating_range):
    from .operators import LOGIC_OPERATORS
    operator, operating_range = '=', np.asarray(operating_range)
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
                    return accumulator(operating_range[b])
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
    type_id, operator = _get_type_id(condition), LOGIC_OPERATORS[operator]

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
        return accumulator(operating_range[b])
    except FoundError as ex:
        return ex.err


_xfilter = np.vectorize(_xfilter, otypes=[object], excluded={0, 1, 3})


def xfilter(accumulator, test_range, condition, operating_range=None):
    operating_range = test_range if operating_range is None else operating_range
    # noinspection PyTypeChecker
    test_range = {'raw': replace_empty(test_range, '')}
    res = _xfilter(accumulator, test_range, condition, operating_range)
    return res.view(Array)


def flatten(v, check=is_number):
    if isinstance(v, np.ndarray):
        if not check:
            yield from v.ravel()
        else:
            yield from filter(check, v.ravel())
    elif not isinstance(v, str) and isinstance(v, Iterable):
        for el in v:
            yield from flatten(el, check)
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


def wrap_ufunc(
        func: t.Callable[..., t.Any], input_parser=lambda *a: map(float, a), check_error: t.Callable[..., t.Any] = get_error,
        args_parser=lambda *a: map(replace_empty, a), otype=Array,
        ranges: bool = False, return_func=lambda res, *args: res, check_nan: bool = True, **kw):
    """A Wraper for numpy's ufunc.
    Helps call a numpy universal function (ufunc)

    Args:
        func (t.Callable[..., t.Any]): _description_
        input_parser (_type_, optional): _description_. Defaults to lambda*a:map(float, a).
        check_error (t.Callable[..., t.Any], optional): _description_. Defaults to get_error.
        args_parser (_type_, optional): _description_. Defaults to lambda*a:map(replace_empty, a).
        otype (_type_, optional): _description_. Defaults to Array.
        ranges (bool, optional): _description_. Defaults to False.
        return_func (_type_, optional): _description_. Defaults to lambdares.
        check_nan (bool, optional): _description_. Defaults to True.

    Raises:
        BroadcastError: _description_
        ex: _description_

    Returns:
        _type_: _description_
    """
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
def get_functions() -> t.Dict[str, t.Callable]:
    """Get's a dictionary with all functions.
    Can be used to create and set custom funcions to be used in formulas.

    Example::

        ..codeblock :: python

            >>> import formulas
            
            >>> FUNCTIONS = formulas.get_functions()
            >>> FUNCTIONS['MY_CUSTOM_FUNC'] = lambda x, y: 1 + y + x
            
            >>> func = formulas.Parser().compile('=MY_CUSTOM_FUNC(1, 2)')
            >>> result = func(4)
            >>> assert result == 4
            True


    Returns:
        t.Dict[str, t.Callable]: The FUNCTIONS dictionary.
    """    
    functions: t.Dict[str, t.Callable] = collections.defaultdict(lambda: not_implemented)
    for name in SUBMODULES:
        functions.update(importlib.import_module(name, __name__).FUNCTIONS)
    functions.update(FUNCTIONS)
    return functions
