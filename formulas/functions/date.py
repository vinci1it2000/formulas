#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2025 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
Python equivalents of financial Excel functions.
"""

import calendar
import collections
import datetime
import functools
import math
import re

import numpy as np
import schedula as sh
from dateutil.relativedelta import relativedelta

from . import (
    COMPILING,
    Error,
    FoundError,
    _convert2float,
    _get_single_args,
    flatten,
    get_error,
    is_number,
    raise_errors,
    replace_empty,
    text2num,
    wrap_func,
    wrap_impure_func,
    wrap_ufunc,
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
        raise FoundError(err=Error.errors["#NUM!"])
    return y, m, d


@functools.lru_cache
def xdate(year, month, day):
    if year == 1900 and (month, day) in ((2, 29), (3, 0)):
        return 60
    d = _date(year + 1900 if year < 1900 else year, month, day)
    return (datetime.datetime(*d) - DATE_ZERO).days + int(d >= (1900, 3, 1))


FUNCTIONS["DATE"] = wrap_ufunc(
    xdate, input_parser=lambda *a: map(lambda v: math.floor(float(v)), a)
)


@functools.lru_cache(None)
def _get_date_parser():
    from dateutil.parser import parser, parserinfo

    info = parserinfo()
    info._year = 1930 + 50
    info._century = 1900
    return parser(info)


@functools.lru_cache
def _text2datetime(date_text):
    res = _get_date_parser()._parse(date_text)[0]
    assert res
    date = [
        getattr(res, v) for v in ("year", "month", "day", "hour", "minute", "second")
    ]
    assert not all(v is None for v in date)
    return tuple(d if v is None else v for v, d in zip(date, DEFAULT_DATE))


def xdatevalue(date_text):
    return xdate(*_text2datetime(date_text)[:3])


FUNCTIONS["DATEVALUE"] = wrap_ufunc(xdatevalue, input_parser=lambda *a: a)


def _int2date(serial_number):
    if 60 < serial_number <= 2958465:
        serial_number -= 1
    elif serial_number == 60:
        return 1900, 2, 29
    elif serial_number == 0:
        return 1900, 1, 0
    elif not 0 < serial_number < 60:
        raise FoundError(err=Error.errors["#NUM!"])

    date = DATE_ZERO + datetime.timedelta(days=serial_number)
    return date.year, date.month, date.day


def xday(serial_number, n=2):
    try:
        serial_number = math.floor(float(serial_number))
        return _int2date(serial_number)[n]
    except ValueError:
        return _text2datetime(serial_number)[n]


FUNCTIONS["DAY"] = wrap_ufunc(functools.partial(xday, n=2), input_parser=lambda *a: a)
FUNCTIONS["MONTH"] = wrap_ufunc(functools.partial(xday, n=1), input_parser=lambda *a: a)
FUNCTIONS["YEAR"] = wrap_ufunc(functools.partial(xday, n=0), input_parser=lambda *a: a)


def xweekday(serial_number, n=1):
    n, serial_number, zero = int(n), int(serial_number), 7
    if not (0 <= serial_number <= 2958465):
        return Error.errors["#NUM!"]
    if 1 <= n <= 2:
        n = n - 1
    elif n == 3:
        n, zero = 2, 0
    elif 11 <= n <= 17:
        n = n - 10
    else:
        return Error.errors["#NUM!"]
    return int(serial_number + 7 - n) % 7 or zero


FUNCTIONS["WEEKDAY"] = wrap_ufunc(xweekday, input_parser=lambda *a: text2num(a))


def xisoweeknum(serial_number):
    if serial_number <= 1:
        return 52
    if serial_number <= 60:
        serial_number -= 1
    return datetime.datetime(*_int2date(serial_number)).isocalendar()[1]


FUNCTIONS["_XLFN.ISOWEEKNUM"] = FUNCTIONS["ISOWEEKNUM"] = wrap_ufunc(
    xisoweeknum, input_parser=lambda *a: text2num(a)
)


def xweeknum(serial_number, n=1):
    args = serial_number, n
    raise_errors(*args)
    args = _get_single_args(*(replace_empty(v) for v in args))
    serial_number, n = (int(text2num(v)) for v in args)
    if not (0 <= serial_number <= 2958465):
        return Error.errors["#NUM!"]
    if 11 <= n <= 17:
        n = n - 9
    elif n == 21:
        return xisoweeknum(serial_number)
    elif not 1 <= n <= 2:
        return Error.errors["#NUM!"]
    n += 7
    return (
        math.floor((serial_number - n) / 7)
        - math.floor((xdate(_int2date(serial_number)[0], 1, 1) - n) / 7)
        + 1
    )


FUNCTIONS["WEEKNUM"] = wrap_func(xweeknum)


def xdatedif(start_date, end_date, unit):
    start_date, end_date = int(start_date), int(end_date)
    if start_date > end_date:
        return Error.errors["#NUM!"]
    if unit == "D":
        if any(not 0 <= v <= 2958465 for v in (start_date, end_date)):
            return Error.errors["#NUM!"]
        return end_date - start_date
    start, end = _int2date(start_date), _int2date(end_date)
    unit = unit.upper()
    if unit == "Y":
        return end[0] - start[0] - int(end[1:] < start[1:])
    if unit == "M":
        r = (end[0] - start[0]) * 12 + end[1] - start[1]
        return r - int(end[2] < start[2])
    if unit in ("MD", "YD"):
        args = list(start)
        i = 2 if unit == "MD" else 1
        if end[i:] < start[i:]:
            args[i - 1] += 1
            if args[1] > 12:
                args[0] += 1
                args[1] = 1
                end = end[0], end[1], end[2] - 1
        return xdate(*args[:i], *end[i:]) - start_date
    if unit == "YM":
        dm = (end[1] - start[1]) - int(end[2] < start[2])
        return dm + int(dm < 0) * 12
    return Error.errors["#NUM!"]


FUNCTIONS["DATEDIF"] = wrap_ufunc(xdatedif, input_parser=lambda *a: text2num(a))


def _xedate(start_date, months):
    raise_errors(start_date, months)
    args = [start_date, text2num(months)]
    for i, v in enumerate(args):
        v = tuple(flatten(v, None))
        if len(v) != 1 or isinstance(v[0], bool):
            return Error.errors["#VALUE!"]
        args[i] = v[0]
    start_date, months = [0 if v is sh.EMPTY else v for v in args]
    if not is_number(months):
        raise FoundError(err=Error.errors["#VALUE!"])
    months = math.trunc(float(months))
    if isinstance(start_date, str):
        date = _text2datetime(start_date)[:3]
    else:
        date = _int2date(int(start_date))
    return date, months


def xedate(start_date, months):
    date, months = _xedate(start_date, months)
    dt = 0
    if date == (1900, 1, 0):
        date = 1900, 1, 1
        dt = 1
    date = datetime.datetime(*date) + relativedelta(months=months)  # type: ignore[arg-type]
    return xdate(date.year, date.month, date.day) - dt


FUNCTIONS["EDATE"] = wrap_func(xedate)


def xeomonth(start_date, months):
    date, months = _xedate(start_date, months)
    if date == (1900, 1, 0):
        date = 1900, 1, 1
    date = datetime.datetime(*date) + relativedelta(months=months + 1, day=1)  # type: ignore[arg-type]
    if date.year < 1900:
        return Error.errors["#NUM!"]
    return xdate(date.year, date.month, date.day) - 1


FUNCTIONS["EOMONTH"] = wrap_func(xeomonth)


def xtoday():
    date = datetime.datetime.now()
    return xdate(date.year, date.month, date.day)


FUNCTIONS["TODAY"] = {
    "extra_inputs": collections.OrderedDict([(COMPILING, False)]),
    "function": wrap_impure_func(wrap_func(xtoday)),
}


def xtime(hour, minute, second):
    if all(x <= 32767 for x in (hour, minute, second)):
        v = hour / 24 + minute / 1440 + second / 86400
        if v >= 0:
            return v % 1
    return Error.errors["#NUM!"]


FUNCTIONS["TIME"] = wrap_ufunc(
    xtime, input_parser=lambda *a: map(lambda v: math.floor(float(v)), a)
)


def xtimevalue(time_text):
    return xtime(*_text2datetime(time_text)[3:])


FUNCTIONS["TIMEVALUE"] = wrap_ufunc(xtimevalue, input_parser=lambda *a: a)


def _n2time(serial_number):
    if serial_number < 0:
        raise FoundError(err=Error.errors["#NUM!"])
    at_hours = (serial_number + 1 / 864e8) * 24
    hours = math.floor(at_hours)
    at_mins = (at_hours - hours) * 60
    mins = math.floor(at_mins)
    secs = (at_mins - mins) * 60
    return hours % 24, mins, int(round(secs - 1.1e-6, 0))


def xsecond(serial_number, n=2):
    try:
        return _n2time(float(serial_number))[n]
    except ValueError:
        return _text2datetime(serial_number)[3 + n]
    except FoundError as ex:
        return ex.err


FUNCTIONS["SECOND"] = wrap_ufunc(
    functools.partial(xsecond, n=2), input_parser=lambda *a: a
)
FUNCTIONS["MINUTE"] = wrap_ufunc(
    functools.partial(xsecond, n=1), input_parser=lambda *a: a
)
FUNCTIONS["HOUR"] = wrap_ufunc(
    functools.partial(xsecond, n=0), input_parser=lambda *a: a
)


def xnow():
    d = datetime.datetime.now()
    return xdate(d.year, d.month, d.day) + xtime(d.hour, d.minute, d.second)


FUNCTIONS["NOW"] = {
    "extra_inputs": collections.OrderedDict([(COMPILING, False)]),
    "function": wrap_impure_func(wrap_func(xnow)),
}


def xyearfrac(start_date, end_date, basis=0):
    raise_errors(basis, start_date, end_date)
    basis = tuple(flatten(basis, None))
    if len(basis) != 1 or isinstance(basis[0], bool):
        return Error.errors["#VALUE!"]
    basis = 0 if basis[0] is sh.EMPTY else basis[0]
    if not is_number(basis) or int(basis) not in (0, 1, 2, 3, 4):
        return Error.errors["#NUM!"]
    dates = [tuple(flatten(d, None)) for d in (start_date, end_date)]
    if any(isinstance(d[0], bool) for d in dates):
        return Error.errors["#VALUE!"]
    # noinspection PyTypeChecker
    basis, dates = int(basis), sorted([xday(*d, slice(0, 3)) for d in dates])  # type: ignore[too-many-positional-arguments]

    return day_count(*dates, basis=basis) / year_days(*dates, basis=basis)


def year_days(start_date, end_date, basis):
    if basis in (0, 2, 4):  # US 30/360 & Actual/360 & Eurobond 30/360
        return 360
    elif basis == 3:  # Actual/365
        return 365
    elif basis == 1:  # Actual/actual
        y1 = start_date[0]
        y2 = end_date[0]
        return 365 + calendar.leapdays(y1, y2 + 1) / (y2 - y1 + 1)


def day_count(start_date, end_date, basis, exact=False):
    y1, m1, d1 = start_date
    y2, m2, d2 = end_date
    if basis in (0, 4):  # US 30/360 & Eurobond 30/360
        d1 = min(d1, 30)
        if basis == 4 or d1 == 30:
            d2 = min(d2, 30)
        elif basis == 0 and d1 in (28, 29):
            if calendar.monthrange(y1, m1)[1] == d1:
                d1 = 30
        n_days = 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)
    else:  # Actual/actual & Actual/360 & Actual/365
        n_days = xdate(y2, m2, d2) - xdate(y1, m1, d1)
    if exact and end_date >= (1900, 3, 1) > start_date:
        n_days -= 1
    return n_days


FUNCTIONS["YEARFRAC"] = wrap_func(xyearfrac)


def xdays(end_date, start_date):
    raise_errors(end_date)
    end_date = xday(end_date, slice(0, 3))
    raise_errors(start_date)
    start_date = xday(start_date, slice(0, 3))
    return day_count(start_date, end_date, 1)


def xdays360(start_date, end_date, method=False):
    raise_errors(start_date)
    start_date = xday(start_date, slice(0, 3))
    raise_errors(end_date)
    end_date = xday(end_date, slice(0, 3))
    raise_errors(method)
    return day_count(start_date, end_date, 4 if _convert2float(method) else 0)


FUNCTIONS["_XLFN.DAYS"] = FUNCTIONS["DAYS"] = wrap_ufunc(
    xdays, input_parser=lambda *a: a, check_error=lambda *a: None
)
FUNCTIONS["_XLFN.DAYS360"] = FUNCTIONS["DAYS360"] = wrap_ufunc(
    xdays360, input_parser=lambda *a: a, check_error=lambda *a: None
)


def xnetworkdays_intl(start_date, end_date, weekend=1, holidays=()):
    raise_errors(start_date)
    start_date = datetime.date(*xday(start_date, slice(0, 3)))
    raise_errors(end_date)
    end_date = datetime.date(*xday(end_date, slice(0, 3)))
    if isinstance(weekend, (int, float, bool)):
        try:
            wm = {
                1: "1111100",  # Sat, Sun
                2: "0111110",  # Sun, Mon
                3: "0011111",  # Mon, Tue
                4: "1001111",  # Tue, Wed
                5: "1100111",  # Wed, Thu
                6: "1110011",  # Thu, Fri
                7: "1111001",  # Fri, Sat
                11: "1111110",  # Sun only
                12: "0111111",  # Mon only
                13: "1011111",  # Tue only
                14: "1101111",  # Wed only
                15: "1110111",  # Thu only
                16: "1111011",  # Fri only
                17: "1111101",  # Sat only
            }[int(weekend)]
        except KeyError:
            raise FoundError(err=Error.errors["#NUM!"]) from None

    elif isinstance(weekend, str) and re.match("[01]{7}", weekend):
        wm = "".join("1" if v == "0" else "0" for v in weekend)
    else:
        raise FoundError(err=Error.errors["#VALUE!"])

    if start_date <= end_date:
        end_date += relativedelta(days=1)
    else:
        end_date -= relativedelta(days=1)
    return np.busday_count(start_date, end_date, weekmask=wm, holidays=holidays)


def args_xnetworkdays_intl(
    start_date, end_date, weekend=1, holidays=(), true_err=False
):
    holidays = set(flatten([replace_empty(holidays)], None))
    if any(isinstance(v, bool) for v in holidays):
        raise FoundError(err=Error.errors["#VALUE!"])

    err = get_error(holidays)
    if err:
        raise FoundError(err=err if true_err else Error.errors["#VALUE!"])

    holidays = {xday(v, slice(0, 3)) for v in holidays}
    return (
        replace_empty(start_date),
        replace_empty(end_date),
        replace_empty(weekend),
        tuple(
            datetime.date(1900, 1, 1) if v == (1900, 1, 0) else datetime.date(*v)
            for v in holidays
        ),
    )


FUNCTIONS["NETWORKDAYS.INTL"] = wrap_ufunc(
    xnetworkdays_intl,
    input_parser=lambda *a: a,
    check_error=lambda *a: None,
    args_parser=args_xnetworkdays_intl,
    excluded={3},
)
FUNCTIONS["NETWORKDAYS"] = wrap_ufunc(
    xnetworkdays_intl,
    input_parser=lambda *a: a,
    check_error=lambda *a: None,
    args_parser=lambda start_date, end_date, hdays=(): args_xnetworkdays_intl(
        start_date, end_date, 1, hdays, True
    ),
    excluded={3},
)


def args_xworkday_intl(start_date, days, weekend=1, holidays=(), true_err=False):
    holidays = set(flatten([replace_empty(holidays)], None))
    if any(isinstance(v, bool) for v in holidays):
        raise FoundError(err=Error.errors["#VALUE!"])

    err = get_error(holidays)
    if err:
        raise FoundError(err=err if true_err else Error.errors["#NUM!"])
    holidays = {xday(v, slice(0, 3)) for v in holidays}
    return (
        replace_empty(start_date),
        replace_empty(days),
        replace_empty(weekend),
        tuple(
            datetime.date(1900, 1, 1) if v == (1900, 1, 0) else datetime.date(*v)
            for v in holidays
        ),
    )


def xworkday_intl(start_date, days, weekend=1, holidays=()):
    raise_errors(start_date)
    start_date = datetime.date(*xday(start_date, slice(0, 3)))
    raise_errors(days)
    days = _convert2float(days)
    if isinstance(weekend, (int, float, bool)):
        try:
            wm = {
                1: "1111100",  # Sat, Sun
                2: "0111110",  # Sun, Mon
                3: "0011111",  # Mon, Tue
                4: "1001111",  # Tue, Wed
                5: "1100111",  # Wed, Thu
                6: "1110011",  # Thu, Fri
                7: "1111001",  # Fri, Sat
                11: "1111110",  # Sun only
                12: "0111111",  # Mon only
                13: "1011111",  # Tue only
                14: "1101111",  # Wed only
                15: "1110111",  # Thu only
                16: "1111011",  # Fri only
                17: "1111101",  # Sat only
            }[int(weekend)]
        except KeyError:
            raise FoundError(err=Error.errors["#NUM!"]) from None

    elif isinstance(weekend, str) and not get_error(weekend):
        if re.match("[01]{7}", weekend):
            wm = "".join("1" if v == "0" else "0" for v in weekend)
        else:
            raise FoundError(err=Error.errors["#VALUE!"])
    else:
        raise FoundError(err=Error.errors["#NUM!"])
    date = np.busday_offset(start_date, days, weekmask=wm, holidays=holidays).astype(
        "O"
    )
    return xdate(date.year, date.month, date.day)


FUNCTIONS["WORKDAY.INTL"] = wrap_ufunc(
    xworkday_intl,
    input_parser=lambda *a: a,
    check_error=lambda *a: None,
    args_parser=args_xworkday_intl,
    excluded={3},
)
FUNCTIONS["WORKDAY"] = wrap_ufunc(
    xworkday_intl,
    input_parser=lambda *a: a,
    check_error=lambda *a: None,
    args_parser=lambda start_date, days, hdays=(): args_xworkday_intl(
        start_date, days, 1, hdays, True
    ),
    excluded={3},
)
