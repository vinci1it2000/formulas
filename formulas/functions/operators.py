#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
Python equivalents of excel operators.
"""

import collections
from . import replace_empty, not_implemented, wrap_func

OPERATORS = collections.defaultdict(lambda: not_implemented)
# noinspection PyTypeChecker
OPERATORS.update({k: wrap_func(v, ranges=k in ' ,:') for k, v in {
    '+': lambda x, y: replace_empty(x) + replace_empty(y),
    '-': lambda x, y: replace_empty(x) - replace_empty(y),
    'U-': lambda x: -replace_empty(x),
    'U+': lambda x: replace_empty(x),
    '*': lambda x, y: replace_empty(x) * replace_empty(y),
    '/': lambda x, y: replace_empty(x) / replace_empty(y),
    '^': lambda x, y: replace_empty(x) ** replace_empty(y),
    '<': lambda x, y: x < y,
    '<=': lambda x, y: x <= y,
    '>': lambda x, y: x > y,
    '>=': lambda x, y: x >= y,
    '=': lambda x, y: x == y,
    '<>': lambda x, y: x != y,
    '&': lambda x, y: replace_empty(x, '') + replace_empty(y, ''),
    '%': lambda x: replace_empty(x) / 100.0,
    ',': lambda x, y: x | y,
    ' ': lambda x, y: x & y,
    ':': lambda x, y: x + y
}.items()})
