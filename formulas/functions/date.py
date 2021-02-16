#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2021 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
Python equivalents of financial Excel functions.
"""
import math
import calendar
import datetime
import functools
import collections
import schedula as sh
from . import (
    wrap_ufunc, Error, FoundError, get_error, wrap_func, raise_errors, flatten,
    is_number, COMPILING, wrap_impure_func
)

FUNCTIONS = {}
DATE_ZERO = datetime.datetime(1899, 12, 31)
DEFAULT_DATE = [datetime.datetime.now().year, 1, 1, 0, 0, 0]


def _date(y, m, d):
    dy = math.floor((m - 1) / 12)
    y, m = y + dy, m - dy * 12

    if d <= 0:
        m -= 1
        d += calendar.monthrange(y, m or 12)[1] + int(y == 1900 and m == 2)
        y, m, d = _date(y, m, d)
    elif y <= 9999:
        max_d = calendar.monthrange(y, m)[1]
        if d > max_d:
            y, m, d = _date(y, m + 1, d - max_d)
    if not (1899 < y <= 9999 or (y, m, d) == (1899, 12, 31)):
        raise FoundError(err=Error.errors['#NUM!'])
    return y, m, d


def xdate(year, month, day):
    d = _date(year + 1900 if year < 1900 else year, month, day)
    return (datetime.datetime(*d) - DATE_ZERO).days + int(d > (1900, 3, 1))


FUNCTIONS['DATE'] = wrap_ufunc(
    xdate, input_parser=lambda *a: map(lambda v: math.floor(float(v)), a)
)


@functools.lru_cache(None)
def _get_date_parser():
    from dateutil.parser import parser, parserinfo
    info = parserinfo()
    info._year = 1930 + 50
    info._century = 1900
    return parser(info)


def _text2datetime(date_text):
    res = _get_date_parser()._parse(date_text)[0]
    assert res
    date = [getattr(res, v) for v in (
        "year", "month", "day", "hour", "minute", "second"
    )]
    assert not all(v is None for v in date)
    return tuple(d if v is None else v for v, d in zip(date, DEFAULT_DATE))


def xdatevalue(date_text):
    return xdate(*_text2datetime(date_text)[:3])


FUNCTIONS['DATEVALUE'] = wrap_ufunc(xdatevalue, input_parser=lambda *a: a)


def _int2date(serial_number):
    if 60 < serial_number <= 2958465:
        serial_number -= 1
    elif 0 < serial_number < 60:
        pass
    elif serial_number == 60:
        return 1900, 2, 29
    elif serial_number == 0:
        return 1900, 1, 0
    else:
        raise FoundError(err=Error.errors['#NUM!'])

    date = DATE_ZERO + datetime.timedelta(days=serial_number)
    return date.year, date.month, date.day


def xday(serial_number, n=2):
    try:
        serial_number = math.floor(float(serial_number))
        return _int2date(serial_number)[n]
    except ValueError:
        return _text2datetime(serial_number)[n]
    except FoundError as ex:
        return ex.err


FUNCTIONS['DAY'] = wrap_ufunc(
    functools.partial(xday, n=2), input_parser=lambda *a: a
)
FUNCTIONS['MONTH'] = wrap_ufunc(
    functools.partial(xday, n=1), input_parser=lambda *a: a
)
FUNCTIONS['YEAR'] = wrap_ufunc(
    functools.partial(xday, n=0), input_parser=lambda *a: a
)


def xtoday():
    date = datetime.datetime.now()
    return xdate(date.year, date.month, date.day)


FUNCTIONS['TODAY'] = {
    'extra_inputs': collections.OrderedDict([(COMPILING, False)]),
    'function': wrap_impure_func(wrap_func(xtoday))
}


def xtime(hour, minute, second):
    if all(x <= 32767 for x in (hour, minute, second)):
        v = hour / 24 + minute / 1440 + second / 86400
        if v >= 0:
            return v % 1
    return Error.errors['#NUM!']


FUNCTIONS['TIME'] = wrap_ufunc(
    xtime, input_parser=lambda *a: map(lambda v: math.floor(float(v)), a)
)


def xtimevalue(time_text):
    return xtime(*_text2datetime(time_text)[3:])


FUNCTIONS['TIMEVALUE'] = wrap_ufunc(xtimevalue, input_parser=lambda *a: a)


def _n2time(serial_number):
    if serial_number < 0:
        raise FoundError(err=Error.errors['#NUM!'])
    at_hours = (serial_number + 1 / 864e8) * 24
    hours = math.floor(at_hours)
    at_mins = (at_hours - hours) * 60
    mins = math.floor(at_mins)
    secs = (at_mins - mins) * 60
    return hours % 24, mins, int(round(secs - 1.1E-6, 0))


def xsecond(serial_number, n=2):
    try:
        return _n2time(float(serial_number))[n]
    except ValueError:
        return _text2datetime(serial_number)[3 + n]
    except FoundError as ex:
        return ex.err


FUNCTIONS['SECOND'] = wrap_ufunc(
    functools.partial(xsecond, n=2), input_parser=lambda *a: a
)
FUNCTIONS['MINUTE'] = wrap_ufunc(
    functools.partial(xsecond, n=1), input_parser=lambda *a: a
)
FUNCTIONS['HOUR'] = wrap_ufunc(
    functools.partial(xsecond, n=0), input_parser=lambda *a: a
)


def xnow():
    d = datetime.datetime.now()
    return xdate(d.year, d.month, d.day) + xtime(d.hour, d.minute, d.second)


FUNCTIONS['NOW'] = {
    'extra_inputs': collections.OrderedDict([(COMPILING, False)]),
    'function': wrap_impure_func(wrap_func(xnow))
}


def xyearfrac(start_date, end_date, basis=0):
    raise_errors(basis, start_date, end_date)
    basis = tuple(flatten(basis, None))
    if len(basis) != 1 or isinstance(basis[0], bool):
        return Error.errors['#VALUE!']
    basis = 0 if basis[0] is sh.EMPTY else basis[0]
    if not is_number(basis) or int(basis) not in (0, 1, 2, 3, 4):
        return Error.errors['#NUM!']
    dates = [tuple(flatten(d, None)) for d in (start_date, end_date)]
    if any(isinstance(d[0], bool) for d in dates):
        return Error.errors['#VALUE!']
    # noinspection PyTypeChecker
    basis, dates = int(basis), [xday(*d, slice(0, 3)) for d in dates]
    err = get_error(*dates)
    if err:
        return err

    (y1, m1, d1), (y2, m2, d2) = sorted(dates)
    denom = 360
    if basis in (0, 4):  # US 30/360 & Eurobond 30/360
        d1 = min(d1, 30)
        if basis == 4:
            d2 = min(d2, 30)
        elif d1 == 30:
            d2 = max(d2, 30)
        n_days = 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)
    else:  # Actual/actual & Actual/360 & Actual/365
        n_days = xdate(y2, m2, d2) - xdate(y1, m1, d1)
        if basis == 3:
            denom = 365
        elif basis == 1:
            denom = 365 + calendar.leapdays(y1, y2 + 1) / (y2 - y1 + 1)
    return n_days / denom


FUNCTIONS['YEARFRAC'] = wrap_func(xyearfrac)
