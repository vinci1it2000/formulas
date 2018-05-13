#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
Python equivalents of excel operators.
"""

import collections
from ..formulas.functions import not_implemented, wrap_func, _replace_empty

OPERATORS = collections.defaultdict(lambda: not_implemented)
# noinspection PyTypeChecker
OPERATORS.update({k: wrap_func(v) for k, v in {
    '+': lambda x, y: _replace_empty(x) + _replace_empty(y),
    '-': lambda x, y: _replace_empty(x) - _replace_empty(y),
    'U-': lambda x: -_replace_empty(x),
    'U+': lambda x: _replace_empty(x),
    '*': lambda x, y: _replace_empty(x) * _replace_empty(y),
    '/': lambda x, y: _replace_empty(x) / _replace_empty(y),
    '^': lambda x, y: _replace_empty(x) ** _replace_empty(y),
    '<': lambda x, y: x < y,
    '<=': lambda x, y: x <= y,
    '>': lambda x, y: x > y,
    '>=': lambda x, y: x >= y,
    '=': lambda x, y: x == y,
    '<>': lambda x, y: x != y,
    '&': lambda x, y: _replace_empty(x, '') + _replace_empty(y, ''),
    '%': lambda x: _replace_empty(x) / 100.0,
    ',': lambda x, y: x | y,
    ' ': lambda x, y: x & y,
    ':': lambda x, y: x + y
}.items()})
