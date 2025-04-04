#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2025 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
Python equivalents of engineering Excel functions.
"""
import itertools
import functools
import schedula as sh
from . import wrap_func, flatten, Error, XlError

FUNCTIONS = {}


def _parseX(x):
    x = list(flatten(x, None))
    if len(x) == 1 and not isinstance(x[0], bool):
        x = x[0]
        if isinstance(x, XlError):
            return x
        x = sh.EMPTY is not x and x or '0'
        if isinstance(x, int) or (isinstance(x, float) and x.is_integer()):
            x = x >= 0 and str(int(x)) or x
        if not (not isinstance(x, str) or len(x) > 10):
            return x
        return Error.errors['#NUM!']
    return Error.errors['#VALUE!']


def _parseDEC(x):
    x = list(flatten(x, None))
    if len(x) == 1 and not isinstance(x[0], bool):
        x = x[0]
        if isinstance(x, XlError):
            return x
        try:
            return int(sh.EMPTY is not x and x or 0)
        except ValueError:
            pass
    return Error.errors['#VALUE!']


_xmask = {2: 1 << 9, 8: 1 << 29, 16: 1 << 39}


def _x2dec(x, base=16):
    if isinstance(x, XlError):
        return x
    try:
        x, y = int(x, base), _xmask[base]
        return (x & ~y) - (y & x)
    except ValueError:
        return Error.errors['#NUM!']


_xfunc = {2: bin, 8: oct, 16: hex}


def _dec2x(x, places=None, base=16):
    x = _parseDEC(x)
    if isinstance(x, XlError):
        return x
    y = _xmask[base]
    if -y <= x < y:
        if x < 0:
            x += y << 1
        x = _xfunc[base](int(x))[2:].upper()
        if places is not None:
            places = int(places)
            if places >= len(x):
                return x.zfill(int(places))
        else:
            return x
    return Error.errors['#NUM!']


def hex2dec2bin2oct(function_id, memo):
    dsp = sh.BlueDispatcher(raises=True)

    for k in ('HEX', 'OCT', 'BIN'):
        dsp.add_data(k, filters=[_parseX])

    dsp.add_function(
        function_id='HEX2DEC',
        function=_x2dec,
        inputs=['HEX'],
        outputs=['DEC']
    )

    dsp.add_function(
        function_id='OCT2DEC',
        function=functools.partial(_x2dec, base=8),
        inputs=['OCT'],
        outputs=['DEC']
    )

    dsp.add_function(
        function_id='BIN2DEC',
        function=functools.partial(_x2dec, base=2),
        inputs=['BIN'],
        outputs=['DEC']
    )

    dsp.add_function(
        function_id='DEC2HEX',
        function=_dec2x,
        inputs=['DEC', 'places'],
        outputs=['HEX']
    )

    dsp.add_function(
        function_id='DEC2OCT',
        function=functools.partial(_dec2x, base=8),
        inputs=['DEC', 'places'],
        outputs=['OCT']
    )

    dsp.add_function(
        function_id='DEC2BIN',
        function=functools.partial(_dec2x, base=2),
        inputs=['DEC', 'places'],
        outputs=['BIN']
    )

    i, o = function_id.split('2')

    _func = sh.DispatchPipe(dsp, function_id, [i, 'places'], [o])

    def func(x, places=None):
        return _func.register(memo=memo)(x, places)

    return func

_memo = {}
for k in map('2'.join, itertools.permutations(['HEX', 'OCT', 'BIN', 'DEC'], 2)):
    FUNCTIONS[k] = wrap_func(hex2dec2bin2oct(k, _memo))
