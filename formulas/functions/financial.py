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

import datetime
import functools
import itertools
import sys
from calendar import monthrange
from collections import deque

# pairwise is only available in Python 3.10+
if sys.version_info >= (3, 10):
    from itertools import pairwise
else:
    from itertools import tee

    def pairwise(iterable):
        """Return successive overlapping pairs taken from the input iterable."""
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)


import numpy as np
from dateutil.relativedelta import relativedelta

from ..errors import FoundError
from . import (
    Array,
    Error,
    XlError,
    _convert2float,
    _get_single_args,
    _text2num,
    convert2float,
    flatten,
    get_error,
    is_number,
    raise_errors,
    replace_empty,
    text2num,
    wrap_func,
    wrap_ufunc,
)
from .date import day_count, xdate, xday, year_days

FUNCTIONS = {}


def xdate2date(*date):
    if date == (1900, 1, 0):
        return datetime.date(1899, 12, 31)
    return datetime.date(*date)


def parse_date(date):
    if isinstance(date, bool):
        raise FoundError(err=Error.errors["#VALUE!"])
    return xday(date, slice(0, 3))


def _xcoup_validate(settlement, maturity, frequency, basis=0):
    if settlement >= maturity or frequency not in (1, 2, 4) or not (0 <= basis <= 4):
        raise FoundError(err=Error.errors["#NUM!"])
    return True


def _xcoup(settlement, maturity, frequency, basis=0):
    _xcoup_validate(settlement, maturity, frequency, basis)
    if monthrange(maturity[0], maturity[1])[1] == maturity[2]:
        dt = relativedelta(months=12 // frequency, day=31)
    else:
        dt = relativedelta(months=12 // frequency)
    d = xdate2date(*maturity)
    settlement = xdate2date(*settlement)

    while d > settlement:
        yield _to_date(d)
        d = d - dt
    yield _to_date(d)


def _to_date(x):
    return max((x.year, x.month, x.day), (1900, 1, 0))


def _build_coupon_schedule(issue, first_interest, settlement, frequency, calc_method):
    if monthrange(first_interest[0], first_interest[1])[1] == first_interest[2]:
        dt = relativedelta(months=12 // frequency, day=31)
    else:
        dt = relativedelta(months=12 // frequency)

    dates = [first_interest]
    d = xdate2date(*first_interest)
    sett = xdate2date(*settlement)
    v = d
    if calc_method:
        i = xdate2date(*issue)
    else:
        i = min(d - dt, sett)
    while v > i:
        v = v - dt
        dates.insert(0, _to_date(v))
    dates.insert(0, _to_date(v))

    # go forwards
    while d < sett:
        nxt = d + dt
        dates.append(_to_date(nxt))
        d = nxt
    return dates


def parse_basis(basis, func=int):
    if isinstance(basis, bool):
        raise FoundError(err=Error.errors["#VALUE!"])
    return func(basis)


def xaccrint(
    issue, first_interest, settlement, rate, par, frequency, basis=0, calc_method=1
):
    frequency = int(frequency)
    basis = int(basis)
    _xcoup_validate(issue, settlement, frequency, basis)

    if rate <= 0 or par <= 0:
        raise FoundError(err=Error.errors["#NUM!"])

    periods = list(
        pairwise(
            _build_coupon_schedule(
                issue, first_interest, settlement, frequency, calc_method
            )
        )
    )

    total = 0.0
    if basis in (0, 2, 4):  # US 30/360 & Actual/360 & Eurobond 30/360
        Di = 360 / frequency
    elif basis == 3:  # Actual/365
        Di = 365 / frequency
    if not calc_method:
        ncd, pcd = deque(_xcoup(issue, first_interest, frequency, basis), maxlen=2)
        if pcd < issue < ncd:
            periods.insert(0, (pcd, ncd))

    for start, end in periods:
        s = max(start, issue)
        e = min(end, settlement)
        if e > s:
            if basis == 1:
                Di = day_count(start, end, basis, exact=True)

            total += day_count(s, e, basis=basis, exact=True) / Di
    return float(total * par * rate / frequency)


FUNCTIONS["ACCRINT"] = wrap_ufunc(
    xaccrint,
    input_parser=lambda issue,
    first_interest,
    settlement,
    rate,
    par,
    frequency,
    basis=0,
    calc_method=1: (
        parse_date(issue),
        parse_date(first_interest),
        parse_date(settlement),
        parse_basis(rate, float),
        parse_basis(par, float),
        parse_basis(frequency, float),
        parse_basis(basis),
        calc_method,
    ),
    args_parser=lambda *a: map(replace_empty, a),
)


def xprice(settlement, maturity, rate, yld, redemption, frequency, basis=0):
    coupons = list(pairwise(_xcoup(settlement, maturity, frequency, basis)))
    if rate < 0 or yld < 0 or redemption <= 0:
        raise FoundError(err=Error.errors["#NUM!"])
    c = len(coupons)
    a = day_count(settlement, coupons[-1][0], basis=basis, exact=True)
    day_count(coupons[-1][1], settlement, basis=basis, exact=True)
    b = day_count(coupons[-1][1], coupons[-1][0], basis=basis, exact=True)
    c0 = 100 * rate / frequency
    c2 = a / b - 1
    c1 = 1 + yld / frequency
    r = redemption / c1 ** (c + c2)
    r -= c0 * (b - a) / b
    for k in range(1, c + 1):
        r += c0 / c1 ** (k - 1 + a / b)
    return r


FUNCTIONS["PRICE"] = wrap_ufunc(
    xprice,
    input_parser=lambda settlement,
    maturity,
    rate,
    yld,
    redemption,
    frequency,
    basis=0: (
        parse_date(settlement),
        parse_date(maturity),
        parse_basis(rate, float),
        parse_basis(yld, float),
        parse_basis(redemption, float),
        parse_basis(frequency, float),
        parse_basis(basis),
    ),
    args_parser=lambda *a: map(replace_empty, a),
)


def xpricedisc(settlement, maturity, discount, redemption, basis=0):
    _xcoup_validate(settlement, maturity, 1, basis)
    if discount <= 0 or redemption <= 0:
        raise FoundError(err=Error.errors["#NUM!"])
    DSM = day_count(settlement, maturity, basis, exact=True)
    B = year_days(settlement, maturity, basis)
    return redemption * (1 - discount * DSM / B)


FUNCTIONS["PRICEDISC"] = wrap_ufunc(
    xpricedisc,
    input_parser=lambda settlement, maturity, discount, redemption, basis=0: (
        parse_date(settlement),
        parse_date(maturity),
        parse_basis(discount, float),
        parse_basis(redemption, float),
        parse_basis(basis),
    ),
    args_parser=lambda *a: map(replace_empty, a),
)


def xpricemat(settlement, maturity, issue, rate, yld, basis=0):
    _xcoup_validate(settlement, maturity, 1, basis)
    if rate < 0 or yld < 0:
        raise FoundError(err=Error.errors["#NUM!"])
    DSM = day_count(settlement, maturity, basis, exact=True)
    B = year_days(settlement, maturity, basis)
    DIM = day_count(issue, maturity, basis, exact=True)
    A = day_count(issue, settlement, basis, exact=True)
    res = 100 + (DIM / B * rate * 100)
    res /= 1 + (DSM / B * yld)
    return res - A / B * rate * 100


FUNCTIONS["PRICEMAT"] = wrap_ufunc(
    xpricemat,
    input_parser=lambda settlement, maturity, issue, rate, yld, basis=0: (
        parse_date(settlement),
        parse_date(maturity),
        parse_date(issue),
        parse_basis(rate, float),
        parse_basis(yld, float),
        parse_basis(basis),
    ),
    args_parser=lambda *a: map(replace_empty, a),
)


def xyield(settlement, maturity, rate, pr, redemption, frequency, basis=0):
    coupons = list(pairwise(_xcoup(settlement, maturity, frequency, basis)))

    if rate < 0 or pr <= 0 or redemption <= 0:
        raise FoundError(err=Error.errors["#NUM!"])
    if len(coupons) <= 1:
        DSR = day_count(settlement, maturity, basis=basis, exact=True)
        A = day_count(coupons[-1][1], settlement, basis=basis, exact=True)
        E = day_count(coupons[-1][1], coupons[-1][0], basis=basis, exact=True)
        c0 = rate / frequency
        c1 = pr / 100 + A / E * c0
        return (redemption / 100 + c0 - c1) / c1 * frequency * E / DSR
    c = len(coupons)
    a = day_count(settlement, coupons[-1][0], basis=basis, exact=True)
    day_count(coupons[-1][1], settlement, basis=basis, exact=True)
    b = day_count(coupons[-1][1], coupons[-1][0], basis=basis, exact=True)
    c0 = 100 * rate / frequency
    c2 = a / b - 1
    c3 = c0 * (b - a) / b

    def func(yld):
        c1 = 1 + yld / frequency
        r = redemption / c1 ** (c + c2) - c3
        for k in range(1, c + 1):
            r += c0 / c1 ** (k - 1 + a / b)
        return r - pr

    from scipy.optimize import newton

    try:
        return newton(func, 0, maxiter=100)
    except RuntimeError:
        raise FoundError(err=Error.errors["#NUM!"]) from None


FUNCTIONS["YIELD"] = wrap_ufunc(
    xyield,
    input_parser=lambda settlement,
    maturity,
    rate,
    pr,
    redemption,
    frequency,
    basis=0: (
        parse_date(settlement),
        parse_date(maturity),
        parse_basis(rate, float),
        parse_basis(pr, float),
        parse_basis(redemption, float),
        parse_basis(frequency, float),
        parse_basis(basis),
    ),
    args_parser=lambda *a: map(replace_empty, a),
)


def xyielddisc(settlement, maturity, price, redemption, basis=0):
    _xcoup_validate(settlement, maturity, 1, basis)
    if price <= 0 or redemption <= 0:
        raise FoundError(err=Error.errors["#NUM!"])
    DSM = day_count(settlement, maturity, basis, exact=True)
    B = year_days(settlement, maturity, basis)
    return (redemption / price - 1) / DSM * B


FUNCTIONS["YIELDDISC"] = wrap_ufunc(
    xyielddisc,
    input_parser=lambda settlement, maturity, price, redemption, basis=0: (
        parse_date(settlement),
        parse_date(maturity),
        parse_basis(price, float),
        parse_basis(redemption, float),
        parse_basis(basis),
    ),
    args_parser=lambda *a: map(replace_empty, a),
)


def xyieldmat(settlement, maturity, issue, rate, pr, basis=0):
    _xcoup_validate(settlement, maturity, 1, basis)
    if rate < 0 or pr <= 0:
        raise FoundError(err=Error.errors["#NUM!"])
    Yim = day_count(issue, maturity, basis, exact=True)
    Yim /= year_days(issue, maturity, basis)
    Yis = day_count(issue, settlement, basis, exact=True)
    Yis /= year_days(issue, settlement, basis)
    Ysm = day_count(settlement, maturity, basis, exact=True)
    Ysm /= year_days(settlement, maturity, basis)
    return ((1 + rate * Yim) / (pr / 100 + rate * Yis) - 1) / Ysm


FUNCTIONS["YIELDMAT"] = wrap_ufunc(
    xyieldmat,
    input_parser=lambda settlement, maturity, issue, rate, pr, basis=0: (
        parse_date(settlement),
        parse_date(maturity),
        parse_date(issue),
        parse_basis(rate, float),
        parse_basis(pr, float),
        parse_basis(basis),
    ),
    args_parser=lambda *a: map(replace_empty, a),
)


def xamorlinc(cost, date_purchased, first_period, salvage, period, rate, basis=0):
    period = int(period)
    if not (
        cost > 0
        and date_purchased <= first_period
        and salvage >= 0
        and period >= 0
        and rate > 0
    ):
        raise FoundError(err=Error.errors["#NUM!"])
    cr = cost * rate
    v = day_count(date_purchased, first_period, basis, exact=True)
    v /= year_days(date_purchased, first_period, basis)
    v *= cr

    if period == 0:
        return v
    v = cost - salvage - v
    n = int(v / cr)
    if 1 <= period <= n:
        return cr
    if period == n + 1:
        return v - n * cr
    return 0


FUNCTIONS["AMORLINC"] = wrap_ufunc(
    xamorlinc,
    input_parser=lambda cost,
    date_purchased,
    first_period,
    salvage,
    period,
    rate,
    basis=0: (
        parse_basis(cost, float),
        parse_date(date_purchased),
        parse_date(first_period),
        parse_basis(salvage, float),
        parse_basis(period, float),
        parse_basis(rate, float),
        parse_basis(basis),
    ),
    args_parser=lambda *a: map(replace_empty, a),
)


def xamordegrc(cost, date_purchased, first_period, salvage, period, rate, basis=0):
    period = int(period)
    if not (
        cost > salvage
        and date_purchased <= first_period
        and salvage >= 0
        and period >= 0
        and rate > 0
    ):
        raise FoundError(err=Error.errors["#NUM!"])
    t = int(np.ceil(1 / rate))
    if period >= t:
        return 0
    if t in (3, 4):
        f = 1.5
    elif t in (5, 6):
        f = 2
    elif t > 6:
        f = 2.5
    else:
        raise FoundError(err=Error.errors["#NUM!"])
    r = day_count(date_purchased, first_period, basis, exact=True)
    r /= year_days(date_purchased, first_period, basis)
    r *= f * cost * rate
    cap = cost - salvage
    r_sum = 0
    r = round(min(r, cap))
    for i in range(3, period + 3):
        r_sum += r
        if r_sum > cap:
            return 0
        elif i < t:
            r = f * rate * (cost - r_sum)
        elif i == t:
            r = (cost - r_sum) / 2
    return round(r)


FUNCTIONS["AMORDEGRC"] = wrap_ufunc(
    xamordegrc,
    input_parser=lambda cost,
    date_purchased,
    first_period,
    salvage,
    period,
    rate,
    basis=0: (
        parse_basis(cost, float),
        parse_date(date_purchased),
        parse_date(first_period),
        parse_basis(salvage, float),
        parse_basis(period, float),
        parse_basis(rate, float),
        parse_basis(basis),
    ),
    args_parser=lambda *a: map(replace_empty, a),
)


def xaccrintm(issue, settlement, rate, par, basis=0):
    _xcoup_validate(issue, settlement, 1, basis)
    if isinstance(rate, bool) or isinstance(par, bool):
        raise FoundError(err=Error.errors["#VALUE!"])
    rate = float(rate)
    par = float(par)
    if rate <= 0 or par <= 0:
        raise FoundError(err=Error.errors["#NUM!"])
    dates = issue, settlement
    total = day_count(*dates, basis=basis, exact=True)
    total /= year_days(*dates, basis=basis)
    return float(total * par * rate)


FUNCTIONS["ACCRINTM"] = wrap_ufunc(
    xaccrintm,
    input_parser=lambda issue, settlement, rate, par, basis=0: (
        parse_date(issue),
        parse_date(settlement),
        parse_basis(rate, float),
        parse_basis(par, float),
        parse_basis(basis),
    ),
    args_parser=lambda *a: map(replace_empty, a),
)


def xtbilleq(settlement, maturity, discount):
    settlement = parse_date(replace_empty(np.atleast_2d(settlement)).item())
    maturity = parse_date(replace_empty(np.atleast_2d(maturity)).item())
    discount = parse_basis(
        replace_empty(np.atleast_2d(discount).item()), _convert2float
    )
    raise_errors(settlement, maturity, discount)
    _xcoup_validate(settlement, maturity, 1, 0)

    if not (0 < discount <= 1) or maturity > (
        settlement[0] + 1,
        settlement[1],
        settlement[2],
    ):
        raise FoundError(err=Error.errors["#NUM!"])

    days = day_count(settlement, maturity, basis=1, exact=True)
    if days <= 182:
        res = 365 * discount / (360 - days * discount)
    else:
        v = days / 365
        P = 100.0 * (1.0 - discount * days / 360.0)
        res = (np.sqrt(v * v - (2 * v - 1) * (1 - 100 / P)) - v) / (v - 0.5)

    if res < 0:
        raise FoundError(err=Error.errors["#NUM!"])
    return res


FUNCTIONS["TBILLEQ"] = wrap_func(xtbilleq)


def xtbillprice(settlement, maturity, discount):
    settlement = parse_date(replace_empty(np.atleast_2d(settlement)).item())
    maturity = parse_date(replace_empty(np.atleast_2d(maturity)).item())
    discount = parse_basis(
        replace_empty(np.atleast_2d(discount).item()), _convert2float
    )
    raise_errors(settlement, maturity, discount)
    _xcoup_validate(settlement, maturity, 1, 0)

    if not (0 < discount <= 1) or maturity > (
        settlement[0] + 1,
        settlement[1],
        settlement[2],
    ):
        raise FoundError(err=Error.errors["#NUM!"])

    days = day_count(settlement, maturity, basis=1, exact=True)
    res = 100 * (1 - days * discount / 360)
    if res < 0:
        raise FoundError(err=Error.errors["#NUM!"])
    return res


FUNCTIONS["TBILLPRICE"] = wrap_func(xtbillprice)


def xtbillyield(settlement, maturity, pr):
    settlement = parse_date(replace_empty(np.atleast_2d(settlement)).item())
    maturity = parse_date(replace_empty(np.atleast_2d(maturity)).item())
    pr = parse_basis(replace_empty(np.atleast_2d(pr).item()), _convert2float)
    raise_errors(settlement, maturity, pr)
    _xcoup_validate(settlement, maturity, 1, 0)

    if pr <= 0 or maturity > (settlement[0] + 1, settlement[1], settlement[2]):
        raise FoundError(err=Error.errors["#NUM!"])

    days = day_count(settlement, maturity, basis=1, exact=True)

    return (100 - pr) / pr * (360 / days)


FUNCTIONS["TBILLYIELD"] = wrap_func(xtbillyield)


def xcoupnum(settlement, maturity, frequency, basis=0, *, coupons=()):
    n = -1
    for _ncd in coupons or _xcoup(settlement, maturity, frequency, basis):
        n += 1
    return n


def xcoupncd(settlement, maturity, frequency, basis=0, *, coupons=()):
    ncd = deque(coupons or _xcoup(settlement, maturity, frequency, basis), maxlen=2)[0]
    return xdate(*ncd)


def xcouppcd(settlement, maturity, frequency, basis=0, *, coupons=()):
    pcd = deque(coupons or _xcoup(settlement, maturity, frequency, basis), maxlen=2)[1]
    return xdate(*pcd)


def xcoupdays(settlement, maturity, frequency, basis=0, *, coupons=()):
    if basis == 1:  # Actual/Actual
        ncd, pcd = deque(
            coupons or _xcoup(settlement, maturity, frequency, basis), maxlen=2
        )
        return day_count(pcd, ncd, basis, exact=True)
    _xcoup_validate(settlement, maturity, frequency, basis)
    if basis in (0, 2, 4):  # 30/360 US, Actual/360, 30E/360
        return 360.0 / frequency
    if basis == 3:  # Actual/365
        return 365.0 / frequency


def xcoupdaybs(settlement, maturity, frequency, basis=0, *, coupons=()):
    pcd = deque(coupons or _xcoup(settlement, maturity, frequency, basis), maxlen=2)[1]
    return day_count(pcd, settlement, basis, exact=True)


def xcoupdaysnc(settlement, maturity, frequency, basis=0, *, coupons=()):
    ncd = deque(coupons or _xcoup(settlement, maturity, frequency, basis), maxlen=2)[0]
    return day_count(settlement, ncd, basis or 4)


kw_coup = {
    "input_parser": lambda settlement, maturity, frequency, basis=0: (
        parse_date(settlement),
        parse_date(maturity),
        parse_basis(frequency),
        parse_basis(basis),
    ),
    "check_error": get_error,
    "args_parser": lambda *a: map(replace_empty, a),
}
FUNCTIONS["COUPNUM"] = wrap_ufunc(xcoupnum, **kw_coup)
FUNCTIONS["COUPNCD"] = wrap_ufunc(xcoupncd, **kw_coup)
FUNCTIONS["COUPPCD"] = wrap_ufunc(xcouppcd, **kw_coup)
FUNCTIONS["COUPDAYS"] = wrap_ufunc(xcoupdays, **kw_coup)
FUNCTIONS["COUPDAYBS"] = wrap_ufunc(xcoupdaybs, **kw_coup)
FUNCTIONS["COUPDAYSNC"] = wrap_ufunc(xcoupdaysnc, **kw_coup)


def xduration(
    settlement, maturity, coupon, yld, frequency, basis=0, *, face=100.0, modified=False
):
    coupons = tuple(_xcoup(settlement, maturity, frequency, basis))
    DSC = xcoupdaysnc(settlement, maturity, frequency, basis=basis, coupons=coupons)
    E = xcoupdays(settlement, maturity, frequency, basis=basis, coupons=coupons)
    N = xcoupnum(settlement, maturity, frequency, basis=basis, coupons=coupons)
    df = 1.0 + yld / frequency

    e = min(1.0, DSC / E) + np.arange(N, dtype=float)
    disc = df**e

    cf = np.full(N, coupon * face / frequency, dtype=float)
    cf[-1] += face

    pv = cf / disc
    price = pv.sum()
    if price == 0:
        raise FoundError(err=Error.errors["#DIV/0"])

    t_years = e / frequency
    D_mac = (t_years * pv).sum() / price
    if modified:
        D_mac /= df
    return float(D_mac)


def xpduration(rate, pv, fv):
    if np.any(rate <= 0) or np.any(pv <= 0) or np.any(fv <= 0):
        raise FoundError(err=Error.errors["#NUM!"])
    return np.log(fv / pv) / np.log1p(rate)


def xrri(rate, pv, fv):
    if np.any(rate <= 0) or np.any(pv <= 0) or np.any(fv <= 0):
        raise FoundError(err=Error.errors["#NUM!"])
    return (fv / pv) ** (1 / rate) - 1


kw_duration = {
    "input_parser": lambda settlement, maturity, coupon, yld, frequency, basis=0: (
        parse_date(settlement),
        parse_date(maturity),
        parse_basis(coupon, float),
        parse_basis(yld, float),
        parse_basis(frequency),
        parse_basis(basis),
    ),
    "check_error": get_error,
    "args_parser": lambda *a: map(replace_empty, a),
}

FUNCTIONS["DURATION"] = wrap_ufunc(xduration, **kw_duration)
FUNCTIONS["MDURATION"] = wrap_ufunc(
    functools.partial(xduration, modified=True), **kw_duration
)
FUNCTIONS["_XLFN.PDURATION"] = FUNCTIONS["PDURATION"] = wrap_ufunc(xpduration)
FUNCTIONS["_XLFN.RRI"] = FUNCTIONS["RRI"] = wrap_ufunc(xrri)


def xeffect(nominal_rate, npery):
    nominal_rate = replace_empty(np.atleast_2d(nominal_rate))
    npery = replace_empty(np.atleast_2d(npery))
    if not (nominal_rate.size == npery.size == 1):
        raise FoundError(err=Error.errors["#VALUE!"])
    nominal_rate = parse_basis(nominal_rate.item(), _convert2float)
    npery = int(parse_basis(npery.item(), _convert2float))
    if nominal_rate <= 0 or npery < 1:
        raise FoundError(err=Error.errors["#NUM!"])

    return (1.0 + nominal_rate / npery) ** npery - 1.0


def xnominal(effect_rate, npery):
    effect_rate = replace_empty(np.atleast_2d(effect_rate))
    npery = replace_empty(np.atleast_2d(npery))
    if not (effect_rate.size == npery.size == 1):
        raise FoundError(err=Error.errors["#VALUE!"])
    effect_rate = parse_basis(effect_rate.item(), _convert2float)
    npery = int(parse_basis(npery.item(), _convert2float))
    if effect_rate <= 0 or npery < 1:
        raise FoundError(err=Error.errors["#NUM!"])
    return ((effect_rate + 1.0) ** (1.0 / npery) - 1.0) * npery


FUNCTIONS["EFFECT"] = wrap_func(xeffect)
FUNCTIONS["NOMINAL"] = wrap_func(xnominal)


def args_parser_intrate(settlement, maturity, investment, redemption, basis=0):
    settlement = parse_date(replace_empty(np.asarray(settlement).item()))
    maturity = parse_date(replace_empty(np.asarray(maturity).item()))

    investment = replace_empty(np.asarray(investment).item())
    redemption = replace_empty(np.asarray(redemption).item())
    if isinstance(investment, bool) or isinstance(redemption, bool):
        raise FoundError(err=Error.errors["#VALUE!"])
    investment = _convert2float(investment)
    redemption = _convert2float(redemption)
    if settlement >= maturity:
        raise FoundError(err=Error.errors["#NUM!"])
    if investment <= 0 or redemption <= 0:
        raise FoundError(err=Error.errors["#NUM!"])
    dates = settlement, maturity
    return (redemption - investment) / investment, dates, basis


def xintrate(num, dates, basis=0):
    yf = day_count(*dates, basis=basis) / year_days(*dates, basis=basis)
    return num / yf


FUNCTIONS["INTRATE"] = wrap_ufunc(
    xintrate,
    input_parser=lambda num, dates, basis: (num, dates, parse_basis(basis)),
    args_parser=args_parser_intrate,
    excluded={0, 1},
)


def args_parser_received(settlement, maturity, investment, discount, basis=0):
    settlement = parse_date(replace_empty(np.asarray(settlement).item()))
    maturity = parse_date(replace_empty(np.asarray(maturity).item()))

    investment = replace_empty(np.asarray(investment).item())
    discount = replace_empty(np.asarray(discount).item())
    if isinstance(investment, bool) or isinstance(discount, bool):
        raise FoundError(err=Error.errors["#VALUE!"])
    investment = _convert2float(investment)
    discount = _convert2float(discount)
    if settlement >= maturity:
        raise FoundError(err=Error.errors["#NUM!"])
    if investment <= 0 or discount <= 0:
        raise FoundError(err=Error.errors["#NUM!"])
    dates = settlement, maturity
    return investment, discount, dates, basis


def xreceived(investment, discount, dates, basis=0):
    yf = day_count(*dates, basis=basis) / year_days(*dates, basis=basis)
    return investment / (1 - discount * yf)


FUNCTIONS["RECEIVED"] = wrap_ufunc(
    xreceived,
    input_parser=lambda investment, discount, dates, basis: (
        investment,
        discount,
        dates,
        parse_basis(basis),
    ),
    args_parser=args_parser_received,
    excluded={0, 1, 2},
)


def args_parser_disc(settlement, maturity, pr, redemption, basis=0):
    settlement = parse_date(replace_empty(np.asarray(settlement).item()))
    maturity = parse_date(replace_empty(np.asarray(maturity).item()))

    pr = replace_empty(np.asarray(pr).item())
    redemption = replace_empty(np.asarray(redemption).item())
    if isinstance(pr, bool) or isinstance(redemption, bool):
        raise FoundError(err=Error.errors["#VALUE!"])
    pr = _convert2float(pr)
    redemption = _convert2float(redemption)
    if settlement >= maturity:
        raise FoundError(err=Error.errors["#NUM!"])
    if redemption <= 0 or pr <= 0:
        raise FoundError(err=Error.errors["#NUM!"])
    dates = settlement, maturity
    return (redemption - pr) / redemption, dates, basis


def xdisc(num, dates, basis=0):
    yf = day_count(*dates, basis=basis) / year_days(*dates, basis=basis)
    return num / yf


FUNCTIONS["DISC"] = wrap_ufunc(
    xdisc,
    input_parser=lambda num, dates, basis: (num, dates, parse_basis(basis)),
    args_parser=args_parser_disc,
    excluded={0, 1},
)


def xdb(cost, salvage, life, period, month=12):
    if cost > 0 and salvage >= 0 and life > 0 and period > 0 and 0 < month < 13:
        rate = round(1 - (salvage / cost) ** (1 / life), 3)
        period = int(period)
        month = int(month)
        life = int(life)
        if period > (life if month == 12 else (life + 1)):
            raise FoundError(err=Error.errors["#NUM!"])

        depk = cost * rate * (month / 12.0)
        dep_cum = 0

        for _k in range(2, min(period, life) + 1):
            dep_cum += depk
            base = cost - dep_cum
            depk = base * rate
        if month != 12 and period == (life + 1):
            dep_cum += depk
            base = cost - dep_cum
            depk = base * rate * ((12 - month) / 12.0)
        return depk

    raise FoundError(err=Error.errors["#NUM!"])


FUNCTIONS["DB"] = wrap_ufunc(xdb, input_parser=convert2float)


def xddb(cost, salvage, life, period, factor=2):
    if (
        cost >= 0
        and salvage >= 0
        and salvage <= cost
        and 1 <= period <= life
        and factor > 0
    ):
        float(period) - 1.0  # Start semi-period.
        float(period)  # End semi-period.
        rate = factor / life
        if rate >= 1:
            new_value = 0
            old_value = cost if period == 1 else 0
        else:
            base = 1.0 - rate
            old_value = cost * (base ** (period - 1))
            new_value = cost * (base**period)
        return max(old_value - max(new_value, salvage), 0)

    raise FoundError(err=Error.errors["#NUM!"])


FUNCTIONS["DDB"] = wrap_ufunc(xddb, input_parser=convert2float)


def total_depr(
    cost: float,
    salvage: float,
    life: float,
    period: float,
    factor: float,
    straight_line: bool,
) -> float:
    """
    Cumulative (total) depreciation up to 'period' (can be fractional),
    using double-declining (or general 'factor') balance with optional
    straight-line switchover when SLN gives a higher amount.

    Mirrors the provided JS logic:
      - period can be fractional (e.g., 3.4); only first and last partials are prorated.
      - When straight_line is True, switch from DDB to SLN when SLN > DDB on that step.
      - Caps total depreciation so book value doesn't go below salvage.
    """
    # fractional remainder in the *requested* period
    frac = period - np.floor(period)

    def ddb_depr_amount(tot_depr: float) -> float:
        # Remaining book value
        book = cost - tot_depr
        # Max remaining depreciable
        rem_dep_cap = book - salvage

        # Nominal DDB for a full period
        ddb_amt = book * (factor / life)
        # Cap so we never go past salvage
        return min(ddb_amt, rem_dep_cap)

    def sln_depr_amount(tot_depr: float, a_period: float) -> float:
        # SLN on (remaining book, same salvage, remaining life)
        return (cost - tot_depr - salvage) / (life - a_period)

    if straight_line:

        def _ddb(tot_depr: float, per: float) -> float:
            # Compute current period's full depreciation using the chosen method
            ddb_amt = ddb_depr_amount(tot_depr)
            sln_amt = sln_depr_amount(tot_depr, per)
            depr = sln_amt if ddb_amt < sln_amt else ddb_amt
            new_tot = tot_depr + depr

            # If the requested 'period' is still within the first (0th) partial period
            if np.floor(period) == 0:
                return new_tot * frac

            # If we're at the period just before the (possibly fractional) target period,
            # add a prorated share of the *next* period's depreciation.
            if np.floor(per) == (np.floor(period) - 1):
                ddb_next = ddb_depr_amount(new_tot)
                sln_next = sln_depr_amount(new_tot, per + 1)
                if ddb_next < sln_next:
                    if np.floor(period) == np.floor(life):
                        depr_next = 0.0
                    else:
                        depr_next = sln_next
                else:
                    depr_next = ddb_next

                return new_tot + depr_next * frac

            # Otherwise, keep iterating full periods
            return _ddb(new_tot, per + 1)
    else:

        def _ddb(tot_depr: float, per: float) -> float:
            # Compute current period's full depreciation using the chosen method
            depr = ddb_depr_amount(tot_depr)
            new_tot = tot_depr + depr

            # If the requested 'period' is still within the first (0th) partial period
            if np.floor(period) == 0:
                return new_tot * frac

            # If we're at the period just before the (possibly fractional) target period,
            # add a prorated share of the *next* period's depreciation.
            if np.floor(per) == (np.floor(period) - 1):
                depr_next = ddb_depr_amount(new_tot)
                return new_tot + depr_next * frac

            # Otherwise, keep iterating full periods
            return _ddb(new_tot, per + 1)

    res = _ddb(0.0, 0.0)
    return res


def xvdb(cost, salvage, life, start_period, end_period, factor=2, no_switch=0):
    if (
        cost < 0
        or salvage < 0
        or life <= 0
        or start_period < 0
        or start_period > end_period
        or end_period > life
        and factor < 0
    ):
        raise FoundError(err=Error.errors["#NUM!"])
    no_switch = not bool(no_switch)
    return round(
        total_depr(cost, salvage, life, end_period, factor, no_switch)
        - total_depr(cost, salvage, life, start_period, factor, no_switch),
        16,
    )


FUNCTIONS["VDB"] = wrap_ufunc(xvdb, input_parser=convert2float)


def xoddfprice(
    settlement, maturity, issue, first_coupon, rate, yld, redemption, frequency, basis=0
):
    if (
        not (maturity > first_coupon > settlement > issue)
        or rate < 0
        or yld <= 0
        or redemption <= 0
    ):
        raise FoundError(err=Error.errors["#NUM!"])
    cps_sm = tuple(_xcoup(settlement, maturity, frequency, basis))
    cps_sf = tuple(_xcoup(settlement, first_coupon, frequency, basis))
    e = xcoupdays(settlement, first_coupon, frequency, basis=basis, coupons=cps_sf)
    dfc = day_count(issue, first_coupon, basis=basis, exact=True)
    x = 1 + yld / frequency
    cpt = 100 * rate / frequency

    if dfc < e:
        n = xcoupnum(settlement, maturity, frequency, basis, coupons=cps_sm)
        dsc = day_count(settlement, first_coupon, basis=basis, exact=True)
        a = day_count(issue, settlement, basis=basis, exact=True)
        y = dsc / e
        p3 = x ** (n - 1 + y)
        res = redemption / p3 + cpt * dfc / e / x**y
        res += sum((cpt / x ** (i + y) for i in range(1, n)), 0)
        res -= a / e * cpt
    else:
        cps_if = tuple(_xcoup(issue, first_coupon, frequency, basis))
        dcnl = 0
        anl = 0
        for index, (start, end) in enumerate(pairwise(cps_if[::-1])):
            nl = day_count(start, end, basis=basis, exact=True) if basis == 1 else e
            dci = (
                nl if index else max(0, day_count(issue, end, basis=basis, exact=True))
            )
            a = max(
                0,
                day_count(
                    max(issue, start), min(settlement, end), basis=basis, exact=True
                ),
            )
            dcnl += dci / nl
            anl += a / nl

        if basis in (2, 3):
            ncd = cps_sf[-2]
            dsc = day_count(settlement, ncd, basis=basis, exact=True)
        else:
            pcd = cps_sf[-1]
            dsc = e - day_count(pcd, settlement, basis=basis, exact=True)
        nq = xcoupnum(settlement, first_coupon, frequency, basis, coupons=cps_sf) - 1
        n = xcoupnum(first_coupon, maturity, frequency, basis)
        y = dsc / e
        p3 = x ** (nq + n + y)
        res = redemption / p3 + cpt * dcnl / x ** (nq + y)
        res += sum((cpt / x ** (i + y) for i in range(nq + 1, nq + n + 1)), 0)
        res -= anl * cpt
    return res


FUNCTIONS["ODDFPRICE"] = wrap_ufunc(
    xoddfprice,
    input_parser=lambda settlement,
    maturity,
    issue,
    first_coupon,
    rate,
    yld,
    redemption,
    frequency,
    basis=0: (
        parse_date(settlement),
        parse_date(maturity),
        parse_date(issue),
        parse_date(first_coupon),
        parse_basis(rate, float),
        parse_basis(yld, float),
        parse_basis(redemption, float),
        parse_basis(frequency, float),
        parse_basis(basis),
    ),
    args_parser=lambda *a: map(replace_empty, a),
)


def xoddfyield(
    settlement, maturity, issue, first_coupon, rate, pr, redemption, frequency, basis=0
):
    if (
        not (maturity > first_coupon > settlement > issue)
        or rate < 0
        or pr <= 0
        or redemption <= 0
    ):
        raise FoundError(err=Error.errors["#NUM!"])
    cps_sm = tuple(_xcoup(settlement, maturity, frequency, basis))
    cps_sf = tuple(_xcoup(settlement, first_coupon, frequency, basis))
    e = xcoupdays(settlement, first_coupon, frequency, basis=basis, coupons=cps_sf)
    dfc = day_count(issue, first_coupon, basis=basis, exact=True)

    cpt = 100 * rate / frequency

    if dfc < e:
        n = xcoupnum(settlement, maturity, frequency, basis, coupons=cps_sm)
        dsc = day_count(settlement, first_coupon, basis=basis, exact=True)
        a = day_count(issue, settlement, basis=basis, exact=True)
        y = dsc / e
        cpt_dfc_e = cpt * dfc / e
        trg = a / e * cpt + pr
        n_1_y = n - 1 + y

        def func(yld):
            x = 1 + yld / frequency
            res = redemption / x**n_1_y + cpt_dfc_e / x**y
            res += sum((cpt / x ** (i + y) for i in range(1, n)), 0)
            return res - trg
    else:
        cps_if = tuple(_xcoup(issue, first_coupon, frequency, basis))
        dcnl = 0
        anl = 0
        for index, (start, end) in enumerate(pairwise(cps_if[::-1])):
            nl = day_count(start, end, basis=basis, exact=True) if basis == 1 else e
            dci = (
                nl if index else max(0, day_count(issue, end, basis=basis, exact=True))
            )
            a = max(
                0,
                day_count(
                    max(issue, start), min(settlement, end), basis=basis, exact=True
                ),
            )
            dcnl += dci / nl
            anl += a / nl

        if basis in (2, 3):
            ncd = cps_sf[-2]
            dsc = day_count(settlement, ncd, basis=basis, exact=True)
        else:
            pcd = cps_sf[-1]
            dsc = e - day_count(pcd, settlement, basis=basis, exact=True)
        nq = xcoupnum(settlement, first_coupon, frequency, basis, coupons=cps_sf) - 1
        n = xcoupnum(first_coupon, maturity, frequency, basis)
        y = dsc / e
        trg = anl * cpt + pr
        nqy = nq + y
        nqny = nqy + n
        cpt_dcnl = cpt * dcnl

        def func(yld):
            x = 1 + yld / frequency
            res = redemption / x**nqny + cpt_dcnl / x**nqy
            res += sum((cpt / x ** (i + y) for i in range(nq + 1, nq + n + 1)), 0)
            return res - trg

    from scipy.optimize import newton

    try:
        return newton(func, 0, maxiter=100)
    except RuntimeError:
        raise FoundError(err=Error.errors["#NUM!"]) from None


FUNCTIONS["ODDFYIELD"] = wrap_ufunc(
    xoddfyield,
    input_parser=lambda settlement,
    maturity,
    issue,
    first_coupon,
    rate,
    pr,
    redemption,
    frequency,
    basis=0: (
        parse_date(settlement),
        parse_date(maturity),
        parse_date(issue),
        parse_date(first_coupon),
        parse_basis(rate, float),
        parse_basis(pr, float),
        parse_basis(redemption, float),
        parse_basis(frequency, float),
        parse_basis(basis),
    ),
    args_parser=lambda *a: map(replace_empty, a),
)


def xoddlprice(
    settlement, maturity, last_interest, rate, yld, redemption, frequency, basis=0
):
    _xcoup_validate(settlement, maturity, frequency, basis)
    if last_interest >= settlement or rate < 0 or yld <= 0 or redemption <= 0:
        raise FoundError(err=Error.errors["#NUM!"])
    DCi = day_count(last_interest, maturity, basis=basis, exact=True)
    DCi /= year_days(last_interest, maturity, basis=basis)
    DSCi = day_count(settlement, maturity, basis=basis, exact=True)
    DSCi /= year_days(settlement, maturity, basis=basis)
    Ai = day_count(last_interest, settlement, basis=basis, exact=True)
    Ai /= year_days(last_interest, settlement, basis=basis)
    return (redemption + 100 * DCi * rate) / (DSCi * yld + 1) - 100 * Ai * rate


FUNCTIONS["ODDLPRICE"] = wrap_ufunc(
    xoddlprice,
    input_parser=lambda settlement,
    maturity,
    last_interest,
    rate,
    yld,
    redemption,
    frequency,
    basis=0: (
        parse_date(settlement),
        parse_date(maturity),
        parse_date(last_interest),
        parse_basis(rate, float),
        parse_basis(yld, float),
        parse_basis(redemption, float),
        parse_basis(frequency, float),
        parse_basis(basis),
    ),
    args_parser=lambda *a: map(replace_empty, a),
)


def xoddlyield(
    settlement, maturity, last_interest, rate, pr, redemption, frequency, basis=0
):
    _xcoup_validate(settlement, maturity, frequency, basis)
    if last_interest >= settlement or rate < 0 or pr <= 0 or redemption <= 0:
        raise FoundError(err=Error.errors["#NUM!"])
    DCi = day_count(last_interest, maturity, basis=basis, exact=True)
    DCi /= year_days(last_interest, maturity, basis=basis)
    DSCi = day_count(settlement, maturity, basis=basis, exact=True)
    DSCi /= year_days(settlement, maturity, basis=basis)
    Ai = day_count(last_interest, settlement, basis=basis, exact=True)
    Ai /= year_days(last_interest, settlement, basis=basis)
    return ((redemption + 100 * DCi * rate) / (pr + 100 * Ai * rate) - 1) / DSCi


FUNCTIONS["ODDLYIELD"] = wrap_ufunc(
    xoddlyield,
    input_parser=lambda settlement,
    maturity,
    last_interest,
    rate,
    pr,
    redemption,
    frequency,
    basis=0: (
        parse_date(settlement),
        parse_date(maturity),
        parse_date(last_interest),
        parse_basis(rate, float),
        parse_basis(pr, float),
        parse_basis(redemption, float),
        parse_basis(frequency, float),
        parse_basis(basis),
    ),
    args_parser=lambda *a: map(replace_empty, a),
)


def xdollarde(fractional, denominator):
    fractional = parse_basis(fractional, _convert2float)
    denominator = int(parse_basis(denominator, _convert2float))
    if denominator > 0:
        int_part = np.trunc(fractional)
        frac_part = fractional - int_part
        scale10 = 10 ** np.ceil(np.log10(denominator))
        return int_part + (frac_part * scale10) / denominator
    elif denominator == 0:
        raise FoundError(err=Error.errors["#DIV/0!"])
    raise FoundError(err=Error.errors["#NUM!"])


def xdollarfr(decimal_dollar, fraction):
    decimal_dollar = parse_basis(decimal_dollar, _convert2float)
    fraction = int(parse_basis(fraction, _convert2float))
    if fraction > 0:
        result = int(decimal_dollar)
        result += (
            (decimal_dollar % 1)
            * np.pow(10, -np.ceil(np.log(fraction) / np.log(10)))
            * fraction
        )

        return result

    raise FoundError(err=Error.errors["#NUM!"])


FUNCTIONS["DOLLARDE"] = wrap_ufunc(xdollarde, input_parser=lambda *a: a)
FUNCTIONS["DOLLARFR"] = wrap_ufunc(xdollarfr, input_parser=lambda *a: a)


def _xnpv(values, dates=None, min_date=0):
    err = get_error(dates, values)
    if not err and any(isinstance(v, bool) for v in flatten((dates, values), None)):
        err = Error.errors["#VALUE!"]
    if err:
        return lambda rate: err, None

    values, dates = tuple(map(replace_empty, (values, dates)))

    def _(x):
        return np.array(text2num(replace_empty(x)), float).ravel()

    if dates is None:
        values = _(values)
        t = np.arange(1, values.shape[0] + 1)
    else:
        dates = np.floor(_(dates))
        i = np.argsort(dates)
        values, dates = _(values)[i], dates[i]
        if (
            len(values) != len(dates)
            or (dates <= min_date).any()
            or (dates >= 2958466).any()
        ):
            return lambda rate: Error.errors["#NUM!"], None
        t = (dates - dates[0]) / 365

    def func(rate):
        return (values / np.power(1 + rate, t)).sum()

    t1, tv = t + 1, -t * values

    def dfunc(rate):
        return (tv / np.power(1 + rate, t1)).sum()

    return func, dfunc


def xnpv(rate, values, dates=None):
    with np.errstate(divide="ignore", invalid="ignore"):
        func = _xnpv(values, dates)[0]

        def _(r):
            e = isinstance(r, str) and Error.errors["#VALUE!"]
            return get_error(r, e) or func(r)

        rate = text2num(replace_empty(rate))
        return np.vectorize(_, otypes=[object])(rate).view(Array)


def xxnpv(rate, values, dates):
    rate = np.asarray(rate)
    if rate.size > 1:
        return Error.errors["#VALUE!"]
    raise_errors(rate)
    rate = _text2num(replace_empty(rate).ravel()[0])
    if isinstance(rate, (bool, str)):
        return Error.errors["#VALUE!"]
    if rate <= 0:
        return Error.errors["#NUM!"]

    return xnpv(rate, values, dates)


FUNCTIONS["NPV"] = wrap_func(lambda r, v, *a: xnpv(r, tuple(flatten((v, a)))))
FUNCTIONS["XNPV"] = wrap_func(xxnpv)


def _npf(func, *args, freturn=lambda x: x, **kwargs):
    import numpy_financial as npf

    r = getattr(npf, func)(*args, **kwargs)
    return freturn(r if getattr(r, "shape", True) else r.ravel()[0])


FUNCTIONS["FV"] = wrap_ufunc(
    functools.partial(_npf, "fv"),
    check_error=lambda *args: None,
    input_parser=lambda rate, nper, pmt, pv=0, type=0: convert2float(
        rate, nper, pmt, pv, type
    ),
)


def args_parser_fvschedule(principal, schedule):
    schedule = tuple(flatten(schedule, None, drop_empty=True))
    for v in schedule:
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return principal, get_error(v) or Error.errors["#VALUE!"]
    return replace_empty(principal), np.prod(1 + np.array(schedule))


def xfvschedule(principal, prod):
    principal = parse_basis(principal, _convert2float)
    raise_errors(prod)
    return principal * prod


FUNCTIONS["FVSCHEDULE"] = wrap_ufunc(
    xfvschedule,
    input_parser=lambda *a: a,
    check_error=lambda *args: None,
    args_parser=args_parser_fvschedule,
    excluded={1},
)


def xispmt(rate, per, nper, pv):
    if not nper:
        raise FoundError(err=Error.errors["#DIV/0!"])
    return pv * rate * (per / nper - 1)


FUNCTIONS["ISPMT"] = wrap_ufunc(xispmt, input_parser=convert2float)


def xsln(cost, salvage, life):
    if not life:
        raise FoundError(err=Error.errors["#DIV/0!"])
    return (cost - salvage) / life


def xsyd(cost, salvage, life, per):
    if per <= 0 or per > life:
        raise FoundError(err=Error.errors["#NUM!"])
    return ((cost - salvage) * (life - per + 1) * 2) / (life * (life + 1))


FUNCTIONS["SLN"] = wrap_ufunc(xsln, input_parser=convert2float)
FUNCTIONS["SYD"] = wrap_ufunc(xsyd, input_parser=convert2float)


def xcumipmt(
    rate,
    nper,
    pv,
    start_period,
    end_period,
    type,
    *,
    func=functools.partial(_npf, "ipmt"),
):
    args = rate, nper, pv, start_period, end_period, type
    args = tuple(map(_text2num, _get_single_args(*map(replace_empty, args))))
    raise_errors(*args)
    if any(not isinstance(v, (float, int)) for v in args):
        return Error.errors["#VALUE!"]
    rate, nper, pv, start_period, end_period, type = args
    if (
        rate <= 0
        or nper <= 0
        or pv <= 0
        or start_period < 1
        or end_period < 1
        or start_period > end_period
        or type not in (0, 1)
        or nper < start_period
        or end_period > nper
    ):
        return Error.errors["#NUM!"]
    per = list(range(int(start_period), int(end_period + 1)))
    res = func(rate, per, nper, pv, fv=0, when=type)
    return res[res < 0].sum()


FUNCTIONS["CUMIPMT"] = wrap_func(xcumipmt)
FUNCTIONS["CUMPRINC"] = wrap_func(
    functools.partial(xcumipmt, func=functools.partial(_npf, "ppmt"))
)

_kw = {"input_parser": convert2float}
FUNCTIONS["PV"] = wrap_ufunc(functools.partial(_npf, "pv"), **_kw)
FUNCTIONS["IPMT"] = wrap_ufunc(
    functools.partial(
        _npf,
        "ipmt",
        freturn=lambda x: x > 0 and Error.errors["#NUM!"] or x,
    ),
    **_kw,
)
FUNCTIONS["PMT"] = wrap_ufunc(functools.partial(_npf, "pmt"), **_kw)


def xppmt(rate, per, nper, pv, fv=0, type=0):
    import numpy_financial as npf

    if per < 1 or per >= nper + 1:
        return Error.errors["#NUM!"]
    return npf.ppmt(rate, per, nper, pv, fv=fv, when=type)


FUNCTIONS["PPMT"] = wrap_ufunc(xppmt, **_kw)


def xrate(nper, pmt, pv, fv=0, type=0, guess=0.1):
    with np.errstate(over="ignore"):
        import numpy_financial as npf

        return npf.rate(nper, pmt, pv, fv=fv, when=type, guess=guess, maxiter=1000)


FUNCTIONS["RATE"] = wrap_ufunc(xrate, **_kw)


def xnper(rate, pmt, pv, fv=0, type=0):
    import numpy_financial as npf

    r = npf.nper(rate, pmt, pv, fv=fv, when=type)
    if rate == 0:
        r = np.sign(npf.nper(0.00000001, pmt, pv, fv=fv, when=type)) * np.abs(r)
    return r


FUNCTIONS["NPER"] = wrap_ufunc(xnper, **_kw)


def xirr(values, guess=0.1):
    with np.errstate(divide="ignore", invalid="ignore"):
        import numpy_financial as npf

        res = npf.irr(tuple(flatten(text2num(replace_empty(values)).ravel())))
        res = (not np.isfinite(res)) and Error.errors["#NUM!"] or res

        def _(g):
            e = isinstance(g, str) and Error.errors["#VALUE!"]
            return get_error(g, e) or res

        guess = text2num(replace_empty(guess))
        return np.vectorize(_, otypes=[object])(guess).view(Array)


FUNCTIONS["IRR"] = wrap_func(xirr)


def mirr_args_parser(values, finance_rate, reinvest_rate):
    values = tuple(
        flatten(
            values.ravel(),
            check=lambda v: isinstance(v, XlError)
            or (not isinstance(v, str) and is_number(v)),
        )
    )

    return (values, replace_empty(finance_rate), replace_empty(reinvest_rate))


def xmirr(values, finance_rate, reinvest_rate):
    raise_errors(finance_rate, reinvest_rate, values)
    res = _npf("mirr", values, finance_rate, reinvest_rate)
    if np.isnan(res):
        raise FoundError(err=Error.errors["#DIV/0!"])
    return res


FUNCTIONS["_XLFN.MIRR"] = FUNCTIONS["MIRR"] = wrap_ufunc(
    xmirr,
    args_parser=mirr_args_parser,
    check_error=lambda *a: None,
    input_parser=lambda values, finance_rate, reinvest_rate: (
        (values,) + tuple(convert2float(finance_rate, reinvest_rate))
    ),
    excluded={0},
)


def _newton(f, df, x, tol=0.0000001):
    xmin = tol - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        for _ in range(100):
            dx = f(x) / df(x)
            if not np.isfinite(dx):
                break
            if abs(dx) <= tol:
                return x
            x = max(xmin, x - dx)
    return Error.errors["#NUM!"]


def xxirr(values, dates, x=0.1):
    x = np.asarray(x, object)
    if x.size > 1:
        return Error.errors["#VALUE!"]
    raise_errors(x)
    x = _text2num(replace_empty(x).ravel()[0])
    if isinstance(x, (bool, str)):
        return Error.errors["#VALUE!"]
    if x < 0:
        return Error.errors["#NUM!"]
    f, df = _xnpv(values, dates, min_date=-1)
    if df is None:
        return f(x)
    return _newton(f, df, x)


FUNCTIONS["XIRR"] = wrap_func(xxirr)
