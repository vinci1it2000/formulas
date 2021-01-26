#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2021 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
Python equivalents of Excel operators.
"""
import schedula as sh
import functools
import collections
from . import (
    replace_empty, not_implemented, wrap_func, wrap_ufunc, Error, value_return
)
from .text import _str
from .look import _get_type_id

OPERATORS = collections.defaultdict(lambda: not_implemented)

numeric_wrap = functools.partial(wrap_ufunc, return_func=value_return)

# noinspection PyTypeChecker
OPERATORS.update({k: numeric_wrap(v) for k, v in {
    '+': lambda x, y: x + y,
    '-': lambda x, y: x - y,
    'U-': lambda x: -x,
    '*': lambda x, y: x * y,
    '/': lambda x, y: (x / y) if y else Error.errors['#DIV/0!'],
    '^': lambda x, y: x ** y,
    '%': lambda x: x / 100.0,
}.items()})
OPERATORS['U+'] = wrap_ufunc(
    lambda x: x, input_parser=lambda *a: a, return_func=value_return
)


def logic_input_parser(x, y):
    if x is sh.EMPTY:
        x = '' if isinstance(y, str) else 0
    if y is sh.EMPTY:
        y = '' if isinstance(x, str) else 0
    return (_get_type_id(x), x), (_get_type_id(y), y)


logic_wrap = functools.partial(
    wrap_ufunc, input_parser=logic_input_parser, return_func=value_return,
    args_parser=lambda *a: a
)
LOGIC_OPERATORS = collections.OrderedDict([
    ('>=', lambda x, y: x >= y),
    ('<=', lambda x, y: x <= y),
    ('<>', lambda x, y: x != y),
    ('<', lambda x, y: x < y),
    ('>', lambda x, y: x > y),
    ('=', lambda x, y: x == y),
])
OPERATORS.update({k: logic_wrap(v) for k, v in LOGIC_OPERATORS.items()})
OPERATORS['&'] = wrap_ufunc(
    lambda x, y: x + y, input_parser=lambda *a: map(_str, a),
    args_parser=lambda *a: (replace_empty(v, '') for v in a),
    return_func=value_return
)
OPERATORS.update({k: wrap_func(v, ranges=True) for k, v in {
    ',': lambda x, y: x | y,
    ' ': lambda x, y: x & y,
    ':': lambda x, y: x + y
}.items()})
