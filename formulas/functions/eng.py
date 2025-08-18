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

import json
import math
import itertools
import functools
import numpy as np
import os.path as osp
import schedula as sh
from . import (
    wrap_ufunc, wrap_func, flatten, Error, XlError, raise_errors, replace_empty,
    str2complex
)
from ..errors import FoundError

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


def _bessel(x, n, fn):
    if n < 0:
        return Error.errors['#NUM!']
    res = fn(n, x)
    if np.isinf(res):
        return Error.errors['#NUM!']
    return res


def _parse_x_n(x, n):
    x = np.asarray(x, object).item()
    raise_errors(x)
    n = np.asarray(n, object).item()
    raise_errors(n)
    if isinstance(x, bool) or isinstance(n, bool):
        raise FoundError(err=Error.errors['#VALUE!'])
    return float(replace_empty(x)), int(float(replace_empty(n)))


def xbesseli(x, n):
    from scipy.special import iv
    x, n = _parse_x_n(x, n)
    return _bessel(x, n, iv)


def xbesselj(x, n):
    from scipy.special import jv
    x, n = _parse_x_n(x, n)
    return _bessel(x, n, jv)


def xbesselk(x, n):
    from scipy.special import kv
    x, n = _parse_x_n(x, n)
    if x < 0:
        return Error.errors['#NUM!']
    return _bessel(x, n, kv)


def xbessely(x, n):
    from scipy.special import yv
    x, n = _parse_x_n(x, n)
    if x < 0:
        return Error.errors['#NUM!']
    return _bessel(x, n, yv)


FUNCTIONS['BESSELJ'] = wrap_func(xbesselj)
FUNCTIONS['BESSELI'] = wrap_func(xbesseli)
FUNCTIONS['BESSELK'] = wrap_func(xbesselk)
FUNCTIONS['BESSELY'] = wrap_func(xbessely)

MAX_BITS = 48
MASK = (1 << MAX_BITS) - 1  # 0xFFFFFFFFFFFF


def _to_uint48(x):
    r = int(x)
    if 0 <= r < MASK and x.is_integer():
        return r
    raise FoundError(err=Error.errors['#NUM!'])


def xbitand(x, y):
    return _to_uint48(x) & _to_uint48(y)


def xbitor(x, y):
    """Excel-like BITOR."""
    return _to_uint48(x) | _to_uint48(y)


def xbitxor(x, y):
    """Excel-like BITXOR."""
    return _to_uint48(x) ^ _to_uint48(y)


def xbitlshift(x, shift):
    """Excel-like BITLSHIFT (shift >= 0)."""
    s = int(shift)
    if s < 0:
        return xbitrshift(x, -s)
    if s <= 53:
        return (_to_uint48(x) << s) & MASK
    raise FoundError(err=Error.errors['#NUM!'])


def xbitrshift(x, shift):
    """Excel-like BITRSHIFT (shift >= 0)."""
    s = int(shift)
    if s < 0:
        return xbitlshift(x, -s)
    if s <= 53:
        return (_to_uint48(x) >> s) & MASK
    raise FoundError(err=Error.errors['#NUM!'])


FUNCTIONS['BITAND'] = FUNCTIONS['_XLFN.BITAND'] = wrap_ufunc(xbitand)
FUNCTIONS['BITOR'] = FUNCTIONS['_XLFN.BITOR'] = wrap_ufunc(xbitor)
FUNCTIONS['BITXOR'] = FUNCTIONS['_XLFN.BITXOR'] = wrap_ufunc(xbitxor)
FUNCTIONS['BITLSHIFT'] = FUNCTIONS['_XLFN.BITLSHIFT'] = wrap_ufunc(xbitlshift)
FUNCTIONS['BITRSHIFT'] = FUNCTIONS['_XLFN.BITRSHIFT'] = wrap_ufunc(xbitrshift)


@functools.lru_cache(maxsize=None)
def _units():
    with open(osp.join(osp.dirname(__file__), 'units.json')) as f:
        return json.load(f)


def xconvert(number, from_unit, to_unit):
    num = np.asarray(number, object).item()
    from_unit = np.asarray(from_unit, object).item()
    to_unit = np.asarray(to_unit, object).item()
    raise_errors(num)
    raise_errors(from_unit)
    raise_errors(to_unit)
    UNITS = _units()
    fu = UNITS.get(from_unit, {})
    tu = UNITS.get(to_unit, {})
    system = set(fu).intersection(tu)
    if not system:
        return Error.errors['#N/A']
    system = list(system)[0]
    if isinstance(num, bool):
        return Error.errors['#VALUE!']
    number = float(replace_empty(num))
    if system == 'temperature':
        number *= fu[system]
        # Convert to Kelvin first
        if from_unit in ("C", "cel"):
            number += 273.15
        elif from_unit in ("F", "fah"):
            number = (number + 459.67) * 5.0 / 9.0
        elif from_unit in ("Rank",):
            number *= 5.0 / 9.0
        elif from_unit in ("Reau",):
            number = number * 5.0 / 4.0 + 273.15

        if to_unit in ("C", "cel"):
            number -= 273.15
        elif to_unit in ("F", "fah"):
            number = number * 9.0 / 5.0 - 459.67
        elif to_unit in ("Rank",):
            number /= 5.0 / 9.0
        elif to_unit in ("Reau",):
            number = (number - 273.15) / 5.0 * 4.0
        return number / tu[system]

    return number * (fu[system] / tu[system])


FUNCTIONS['CONVERT'] = FUNCTIONS['_XLFN.CONVERT'] = wrap_func(xconvert)


def _parse_float(x):
    x = replace_empty(np.asarray(x, object).item())
    raise_errors(x)
    if isinstance(x, bool):
        raise FoundError(err=Error.errors['#VALUE!'])
    return float(x)


def xerf_precise(x, func=math.erf):
    return func(_parse_float(x))


def xerf(lower, upper=None):
    res = xerf_precise(lower)
    if upper is not None:
        res = xerf_precise(upper) - res
    return res


FUNCTIONS['ERF'] = FUNCTIONS['_XLFN.ERF'] = wrap_func(xerf)
FUNCTIONS['ERF.PRECISE'] = FUNCTIONS['_XLFN.ERF.PRECISE'] = wrap_func(
    xerf_precise
)
FUNCTIONS['ERFC'] = FUNCTIONS['ERFC.PRECISE'] = wrap_func(functools.partial(
    xerf_precise, func=math.erfc
))
FUNCTIONS['_XLFN.ERFC'] = FUNCTIONS['_XLFN.ERFC.PRECISE'] = FUNCTIONS['ERFC']


def xdelta(x, y=0):
    return 1 if np.isclose(_parse_float(x), _parse_float(y)) else 0


def xgestep(x, step=0):
    return 1 if _parse_float(x) >= _parse_float(step) else 0


FUNCTIONS['DELTA'] = wrap_func(xdelta)
FUNCTIONS['GESTEP'] = wrap_func(xgestep)


def _fmt_complex(r, i, suffix="j"):
    if not (np.isfinite(r) and np.isfinite(i)):
        raise FoundError(err=Error.errors['#NUM!'])
    res = str(complex(r, i)).upper().replace("J", suffix)
    res = res.lstrip('(').rstrip(')')
    if res.endswith(f"+1{suffix}") or res.endswith(f"-1{suffix}"):
        res = res[:-2] + suffix
    elif res.endswith(f"+0{suffix}") or res.endswith(f"-0{suffix}"):
        res = res[:-3]
    elif res in (f"1{suffix}",):
        res = suffix
    elif res in (f"0{suffix}",):
        res = '0'
    if res.startswith("0+") or res.startswith("0-"):
        res = res[2:]
    return res or '0'


def xcomplex(real_num, i_num, suffix="i"):
    if suffix not in ("i", "j"):
        return Error.errors['#VALUE!']
    r = np.asarray(real_num, object).item()
    raise_errors(r)
    if isinstance(r, bool):
        return Error.errors['#VALUE!']
    r = float(replace_empty(r))
    i = np.asarray(i_num, object).item()
    raise_errors(i)
    if isinstance(i, bool):
        return Error.errors['#VALUE!']
    i = float(replace_empty(i))

    return _fmt_complex(r, i, suffix)


FUNCTIONS['COMPLEX'] = FUNCTIONS['_XLFN.COMPLEX'] = wrap_func(xcomplex)


def _parse_im(s):
    s = np.asarray(s, object).item()
    raise_errors(s)
    s = replace_empty(s)
    if isinstance(s, str):
        try:
            return str2complex(s), 'i' if 'i' in s else 'j'
        except ValueError:
            raise FoundError(err=Error.errors['#NUM!'])
    if isinstance(s, complex):
        return s, None
    if isinstance(s, bool):
        pass
    elif isinstance(s, (int, float)):
        return complex(float(s), 0.0), None
    raise ValueError("IM* functions require a string or number.")


def _fmt_im(z, suffix="i"):
    return _fmt_complex(z.real, z.imag, suffix or 'i')


def _xim2num(func, z):
    num = _parse_im(z)[0]
    with np.errstate(divide='ignore', invalid='ignore'):
        return func(num)


def _xim2im(func, z):
    num, suffix = _parse_im(z)
    with np.errstate(divide='ignore', invalid='ignore'):
        return _fmt_im(func(num), suffix)


def _xim2accim(func, initial, *args):
    result = initial
    suffix = None
    for a in flatten(args, check=None):
        num, sfx = _parse_im(a)
        if not suffix:
            suffix = sfx
        elif sfx and suffix != sfx:
            return Error.errors['#VALUE!']
        result = func(result, num)
    return _fmt_im(result, suffix)


def _xyim2im(func, z1, z2):
    num1, sfx1 = _parse_im(z1)
    num2, sfx2 = _parse_im(z2)
    sfx1 = sfx1 or sfx2
    if sfx1 == (sfx2 or sfx1):
        try:
            with np.errstate(divide='ignore', invalid='ignore'):
                return _fmt_im(func(num1, num2), sfx1)
        except ZeroDivisionError:
            pass
    return Error.errors['#NUM!']


FUNCTIONS['IMDIV'] = FUNCTIONS['_XLFN.IMDIV'] = wrap_func(
    functools.partial(_xyim2im, lambda x, y: x / y)
)
FUNCTIONS['IMSUB'] = FUNCTIONS['_XLFN.IMSUB'] = wrap_func(
    functools.partial(_xyim2im, lambda x, y: x - y)
)
FUNCTIONS['IMSUM'] = FUNCTIONS['_XLFN.IMSUM'] = wrap_func(
    functools.partial(_xim2accim, lambda x, y: x + y, complex(0, 0))
)
FUNCTIONS['IMPRODUCT'] = FUNCTIONS['_XLFN.IMPRODUCT'] = wrap_func(
    functools.partial(_xim2accim, lambda x, y: x * y, complex(1, 0))
)
FUNCTIONS['IMABS'] = FUNCTIONS['_XLFN.IMABS'] = wrap_func(
    functools.partial(_xim2num, np.abs)
)
FUNCTIONS['IMREAL'] = FUNCTIONS['_XLFN.IMREAL'] = wrap_func(
    functools.partial(_xim2num, lambda z: z.real)
)
FUNCTIONS['IMAGINARY'] = FUNCTIONS['_XLFN.IMAGINARY'] = wrap_func(
    functools.partial(_xim2num, lambda z: z.imag)
)


def ximargument(x):
    if np.isclose(x.real, 0):
        return Error.errors['#DIV/0!']
    return np.atan2(x.imag, x.real)


FUNCTIONS['IMARGUMENT'] = FUNCTIONS['_XLFN.IMARGUMENT'] = wrap_func(
    functools.partial(_xim2num, ximargument)
)
FUNCTIONS['IMCONJUGATE'] = FUNCTIONS['_XLFN.IMCONJUGATE'] = wrap_func(
    functools.partial(_xim2im, lambda x: x.conjugate())
)
FUNCTIONS['IMCOS'] = FUNCTIONS['_XLFN.IMCOS'] = wrap_func(
    functools.partial(_xim2im, np.cos)
)
FUNCTIONS['IMCOSH'] = FUNCTIONS['_XLFN.IMCOSH'] = wrap_func(
    functools.partial(_xim2im, np.cosh)
)
FUNCTIONS['IMCOT'] = FUNCTIONS['_XLFN.IMCOT'] = wrap_func(
    functools.partial(_xim2im, lambda x: 1 / np.tan(x))
)
FUNCTIONS['IMCSC'] = FUNCTIONS['_XLFN.IMCSC'] = wrap_func(
    functools.partial(_xim2im, lambda x: 1 / np.sin(x))
)
FUNCTIONS['IMCSCH'] = FUNCTIONS['_XLFN.IMCSCH'] = wrap_func(
    functools.partial(_xim2im, lambda x: 1 / np.sinh(x))
)
FUNCTIONS['IMEXP'] = FUNCTIONS['_XLFN.IMEXP'] = wrap_func(
    functools.partial(_xim2im, np.exp)
)
FUNCTIONS['IMLN'] = FUNCTIONS['_XLFN.IMLN'] = wrap_func(
    functools.partial(_xim2im, np.log)
)
FUNCTIONS['IMLOG10'] = FUNCTIONS['_XLFN.IMLOG10'] = wrap_func(
    functools.partial(_xim2im, np.log10)
)
FUNCTIONS['IMLOG2'] = FUNCTIONS['_XLFN.IMLOG2'] = wrap_func(
    functools.partial(_xim2im, np.log2)
)
FUNCTIONS['IMSEC'] = FUNCTIONS['_XLFN.IMSEC'] = wrap_func(
    functools.partial(_xim2im, lambda x: 1 / np.cos(x))
)
FUNCTIONS['IMSECH'] = FUNCTIONS['_XLFN.IMSECH'] = wrap_func(
    functools.partial(_xim2im, lambda x: 1 / np.cosh(x))
)
FUNCTIONS['IMSIN'] = FUNCTIONS['_XLFN.IMSIN'] = wrap_func(
    functools.partial(_xim2im, np.sin)
)
FUNCTIONS['IMSINH'] = FUNCTIONS['_XLFN.IMSINH'] = wrap_func(
    functools.partial(_xim2im, np.sinh)
)
FUNCTIONS['IMSQRT'] = FUNCTIONS['_XLFN.IMSQRT'] = wrap_func(
    functools.partial(_xim2im, np.sqrt)
)
FUNCTIONS['IMTAN'] = FUNCTIONS['_XLFN.IMTAN'] = wrap_func(
    functools.partial(_xim2im, np.tan)
)


def ximpower(z, power):
    num, suffix = _parse_im(z)
    p = replace_empty(np.asarray(power, object).item())
    raise_errors(p)
    return _fmt_im(np.pow(num, float(p)), suffix)


FUNCTIONS['IMPOWER'] = FUNCTIONS['_XLFN.IMPOWER'] = wrap_func(ximpower)
