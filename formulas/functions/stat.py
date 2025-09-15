#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2025 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
Python equivalents of statistical Excel functions.
"""
import math
import functools
import numpy as np
import schedula as sh
from collections import Counter
from . import (
    raise_errors, flatten, wrap_func, Error, is_number, _text2num, xfilter,
    XlError, wrap_ufunc, replace_empty, get_error, is_not_empty, _convert_args,
    convert_nan, FoundError, xfilters, Array
)
from statistics import NormalDist
from scipy import stats, linalg
from scipy.interpolate import interp1d

FUNCTIONS = {}


def _convert(v):
    if isinstance(v, str):
        return 0
    if isinstance(v, bool):
        return int(v)
    if not isinstance(v, (float, int)):
        return float(v)
    return v


def xfunc(*args, func=max, check=is_number, convert=None, default=0,
          _raise=True, parse_args=None):
    _raise and raise_errors(args)
    if parse_args:
        args = parse_args(args)
    it = flatten(map(_convert_args, args), check=check)
    default = [] if default is None else [default]
    return func(list(map(convert, it) if convert else it) or default)


def _xaverage(v):
    if v:
        return np.mean(v)
    return Error.errors['#DIV/0!']


def _xavedev(v):
    if v:
        mu = np.mean(v)
        return np.mean(np.abs(np.asarray(v) - mu))
    return Error.errors['#NUM!']


def _xdevsq(v):
    if v:
        mu = np.mean(v)
        return np.sum((np.asarray(v) - mu) ** 2)
    return Error.errors['#NUM!']


def _xgeomean(v):
    v = np.asarray(v)
    if not v.size or (v <= 0.0).any():
        return Error.errors['#NUM!']

    # log-mean -> exp for numerical stability
    return np.exp(np.mean(np.log(v)))


def _xharmean(v):
    v = np.asarray(v)
    if not v.size or (v <= 0.0).any():
        return Error.errors['#NUM!']
    return v.size / np.sum(1.0 / v)


def _xkurt(v):
    n = len(v)
    if n < 4 or np.isclose(np.var(v, ddof=1), 0):
        return Error.errors['#DIV/0!']

    # SciPy: fisher=True -> excess kurtosis; bias=False -> unbiased (Excel)
    return float(stats.kurtosis(v, fisher=True, bias=False, nan_policy='omit'))


def _xmode_mult(v):
    cnt = Counter(v)
    maxc = max(cnt.values())
    if maxc < 2:
        return Error.errors['#N/A']
    return [k for k, c in cnt.items() if c == maxc]


def _xmode_sngl(v):
    cnt = Counter(v)
    maxc = max(cnt.values())
    if maxc < 2:
        return Error.errors['#N/A']
    # choose smallest among values with count == maxc
    for k, c in cnt.items():
        if c == maxc:
            return k


def _xskew(bias, v):
    n = len(v)
    if n < 3 or np.isclose(np.var(v, ddof=1), 0.0):
        return Error.errors['#DIV/0!']

    return float(stats.skew(
        v, bias=bias, nan_policy='omit'
    ))


xskewp = functools.partial(
    xfunc, func=functools.partial(_xskew, True), default=None
)
FUNCTIONS['_XLFN.SKEW.P'] = FUNCTIONS['SKEW.P'] = wrap_func(xskewp)
xskew = functools.partial(
    xfunc, func=functools.partial(_xskew, False), default=None
)
FUNCTIONS['_XLFN.SKEW'] = FUNCTIONS['SKEW'] = wrap_func(xskew)
xmode_sngl = functools.partial(
    xfunc, func=_xmode_sngl, convert=_convert,
    parse_args=lambda a: (v for v in a if v is not "")
)
FUNCTIONS['_XLFN.MODE.SNGL'] = FUNCTIONS['MODE.SNGL'] = wrap_func(xmode_sngl)
xmode_mult = functools.partial(
    xfunc, func=_xmode_mult, convert=_convert,
    parse_args=lambda a: (v for v in a if v is not "")
)
FUNCTIONS['_XLFN.MODE.MULT'] = FUNCTIONS['MODE.MULT'] = wrap_func(xmode_mult)
xkurt = functools.partial(xfunc, func=_xkurt, default=None)
FUNCTIONS['KURT'] = wrap_func(xkurt)
xharmean = functools.partial(xfunc, func=_xharmean, default=None)
FUNCTIONS['HARMEAN'] = wrap_func(xharmean)
xgeomean = functools.partial(xfunc, func=_xgeomean, default=None)
FUNCTIONS['GEOMEAN'] = wrap_func(xgeomean)
xdevsq = functools.partial(xfunc, func=_xdevsq, default=None)
FUNCTIONS['DEVSQ'] = wrap_func(xdevsq)
xavedev = functools.partial(xfunc, func=_xavedev, default=None)
FUNCTIONS['AVEDEV'] = wrap_func(xavedev)
xaverage = functools.partial(xfunc, func=_xaverage, default=None)
FUNCTIONS['AVERAGE'] = wrap_func(xaverage)
FUNCTIONS['AVERAGEA'] = wrap_func(functools.partial(
    xfunc, convert=_convert, check=is_not_empty, func=_xaverage, default=None
))
FUNCTIONS['AVERAGEIF'] = wrap_func(functools.partial(xfilter, xaverage))
FUNCTIONS['AVERAGEIFS'] = wrap_func(functools.partial(xfilters, xaverage))


def xcorrel(arr1, arr2):
    try:
        arr1, arr2 = _parse_yxp(arr1, arr2)
    except FoundError as ex:
        return ex.err
    return np.corrcoef(arr1, arr2)[0, 1]


FUNCTIONS['CORREL'] = wrap_func(xcorrel)
FUNCTIONS['COUNT'] = wrap_func(functools.partial(
    xfunc, func=len, _raise=False, default=None,
    check=functools.partial(is_number, xl_return=False)
))
FUNCTIONS['COUNTA'] = wrap_func(functools.partial(
    xfunc, check=is_not_empty, func=len, _raise=False, default=None
))
FUNCTIONS['COUNTBLANK'] = wrap_func(functools.partial(
    xfunc, check=lambda x: (x == '' or x is sh.EMPTY), func=len,
    _raise=False, default=None
))
FUNCTIONS['COUNTIF'] = wrap_func(functools.partial(
    xfilter, len, operating_range=None
))
FUNCTIONS['COUNTIFS'] = wrap_func(functools.partial(
    xfilters, len, None
))


def xsort(values, k, large=True):
    err = get_error(k)
    if err:
        return err
    k = float(_text2num(k))
    if isinstance(values, XlError):
        return values
    if 1 <= k <= len(values):
        if large:
            k = -k
        else:
            k -= 1
        return values[math.floor(k)]
    return Error.errors['#NUM!']


def _sort_parser(values, k):
    if isinstance(values, XlError):
        raise FoundError(err=values)
    err = get_error(values)
    if err:
        return err, k
    values = np.array(tuple(flatten(
        values, lambda v: not isinstance(v, (str, bool))
    )), float)
    values.sort()
    return values, replace_empty(k)


FUNCTIONS['LARGE'] = wrap_ufunc(
    xsort, args_parser=_sort_parser, excluded={0}, check_error=lambda *a: None,
    input_parser=lambda *a: a
)

FUNCTIONS['SMALL'] = wrap_ufunc(
    xsort, args_parser=_sort_parser, excluded={0}, check_error=lambda *a: None,
    input_parser=lambda values, k: (values, k, False)
)
FUNCTIONS['MAX'] = wrap_func(xfunc)
FUNCTIONS['MAXA'] = wrap_func(functools.partial(
    xfunc, convert=_convert, check=is_not_empty
))
FUNCTIONS['_XLFN.MAXIFS'] = FUNCTIONS['MAXIFS'] = wrap_func(
    functools.partial(xfilters, xfunc)
)

FUNCTIONS['MEDIAN'] = wrap_func(functools.partial(
    xfunc, func=lambda x: convert_nan(np.median(x) if x else np.nan),
    default=None
))
FUNCTIONS['MIN'] = wrap_func(functools.partial(xfunc, func=min))
FUNCTIONS['MINA'] = wrap_func(functools.partial(
    xfunc, convert=_convert, check=is_not_empty, func=min
))
FUNCTIONS['_XLFN.MINIFS'] = FUNCTIONS['MINIFS'] = wrap_func(functools.partial(
    xfilters, functools.partial(xfunc, func=min)
))


def _forecast_known_filter(known_y, known_x):
    for v in zip(known_y, known_x):
        if not any(isinstance(i, (str, bool)) for i in v):
            yield v


def xslope(yp, xp):
    try:
        a, b = _slope_coeff(*map(np.array, _parse_yxp(yp, xp)))
    except FoundError as ex:
        if ex.err is Error.errors['#NULL!']:
            return Error.errors['#NUM!']
        return ex.err
    return b


def xintercept(yp, xp):
    try:
        a, b = _slope_coeff(*map(np.array, _parse_yxp(yp, xp)))
    except FoundError as ex:
        if ex.err is Error.errors['#NULL!']:
            return Error.errors['#NUM!']
        return ex.err
    return a


FUNCTIONS['SLOPE'] = wrap_func(xslope)
FUNCTIONS['INTERCEPT'] = wrap_func(xintercept)


def _parse_yxp(yp, xp):
    yp, xp = tuple(flatten(yp, check=None)), tuple(flatten(xp, check=None))
    if (sh.EMPTY,) == yp or (sh.EMPTY,) == xp:
        raise FoundError(err=Error.errors['#VALUE!'])
    if len(yp) != len(xp):
        raise FoundError(err=Error.errors['#N/A'])
    raise_errors(*zip(yp, xp))
    yxp = tuple(_forecast_known_filter(yp, xp))
    if len(yxp) <= 1:
        raise FoundError(err=Error.errors['#DIV/0!'])
    return tuple(zip(*yxp))


def _slope_coeff(yp, xp):
    ym, xm = yp.mean(), xp.mean()
    dx = xp - xm
    b = (dx ** 2).sum()
    if not b:
        raise FoundError(err=Error.errors['#DIV/0!'])
    b = (dx * (yp - ym)).sum() / b
    a = ym - xm * b
    return a, b


def _args_parser_forecast(x, yp, xp):
    x = replace_empty(x)
    try:
        a, b = _slope_coeff(*map(np.array, _parse_yxp(yp, xp)))
    except FoundError as ex:
        return x, ex.err
    return x, a, b


def _prepare_ets_data(values, timeline, data_completion, aggregation):
    if data_completion not in (0, 1):
        raise FoundError(err=Error.errors['#NUM!'])
    # 1) parse values
    yp, xp = _parse_yxp(values, timeline)
    yp = np.asarray(yp, float)
    xp = np.asarray(xp, float)

    # 2) sort by timeline
    idx = np.argsort(xp)
    xp, yp = xp[idx], yp[idx]

    # 3) resample time
    x = np.unique(xp)
    diffs = np.diff(x)
    med = np.median(diffs)
    rounded = np.round(diffs / med, 6) * med
    uniq, counts = np.unique(rounded, return_counts=True)
    step = uniq[np.argmax(counts)]
    x0 = [[]]
    for i, diff in enumerate(rounded):
        if diff == step:
            if x0[-1]:
                x0[-1][0] += 1
            else:
                x0[-1] = [1, x[i]]
        elif x0[-1]:
            x0.append([])
    x0 = float(max(x0, key=lambda x: x[0] if x else 0)[1])
    i = np.where(xp == x0)[0][0]
    x0 = x0 - step * int(math.floor((x0 - xp[0]) / step))
    n_steps = int(math.floor((xp[-1] - x0) / step)) + 1
    x = x0 + step * np.arange(n_steps)
    y = np.full_like(x, np.nan, dtype=float)
    idx = np.round((xp - x[0]) / step).astype(int)
    if y.size < 2:
        raise FoundError(err=Error.errors['#NUM!'])
    # 4) aggregate duplicates
    aggr = {
        0: np.mean,
        1: np.sum,
        2: FUNCTIONS['COUNT'],
        3: FUNCTIONS['COUNTA'],
        4: np.min,
        5: np.max,
        6: np.median
    }[int(aggregation)]

    for i in np.unique(idx):
        y[i] = aggr(yp[idx == i])

    # 4) data completion (fill gaps)
    b = np.isnan(y)
    if b.any():
        if data_completion:
            y[b] = interp1d(
                x[~b], y[~b], kind='linear', fill_value="extrapolate"
            )(x[b])
        else:
            y = np.nan_to_num(y, False)
    return x, y, step


def _train_ets_model(
        values, timeline, seasonality, data_completion, aggregation):
    x, y, step = _prepare_ets_data(
        values, timeline, data_completion, aggregation
    )

    import pandas as pd
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    # 5) seasonality
    seas = int(seasonality)
    if seas == 1:  # auto
        seas = 0
        ys = y - y.mean()
        denom = np.dot(ys, ys)
        if denom > 0.0:
            best_p, best_r = 0, 0.0
            for p in range(2, int(min(8760, math.ceil(ys.size / 2.5)))):
                r = np.dot(ys[p:], ys[:-p])
                if r > best_r:
                    best_r, best_p = r, p
            if best_r / denom > 0.5:
                seas = best_p
    elif seas < 0 or seas > 8760:
        raise FoundError(err=Error.errors['#NUM!'])

    kw = {}
    if seas >= 2:
        kw['seasonal'] = 'add'
        kw['seasonal_periods'] = seas
        if y.size < 2 * seas:
            _y = y[:seas]
            kw['initialization_method'] = 'known'
            kw['initial_level'] = np.mean(_y)
            kw['initial_trend'] = (_y[-1] - _y[0]) / _y.size
            kw['initial_seasonal'] = _y - kw['initial_level']
    model = ETSModel(pd.Series(y), trend='add', **kw).fit(
        warn_convergence=False
    )
    return model, x, y, step


def xforecast_ets(
        target_date, values, timeline, seasonality=1,
        data_completion=1, aggregation=0):
    model, x, y, step = _train_ets_model(
        values, timeline, seasonality, data_completion, aggregation
    )
    if target_date <= x[0] - step:
        return Error.errors['#NUM!']

    h = int(math.floor((target_date - x[0]) / 1)) + 1
    h = h or 1
    x = x[0] + step * np.arange(h + 1)
    pred = model.get_prediction(start=0, end=h)
    return float(interp1d(
        x, pred.predicted_mean, kind='linear', fill_value="extrapolate"
    )(target_date))


def xforecast_ets_confint(
        target_date, values, timeline, confidence_level=.95, seasonality=1,
        data_completion=1, aggregation=0):
    model, x, y, step = _train_ets_model(
        values, timeline, seasonality, data_completion, aggregation
    )
    if target_date <= x[0] - step:
        return Error.errors['#NUM!']
    h = (int(math.floor((target_date - x[0]) / 1)) + 1) or 1
    x = x[0] + step * np.arange(h + 1)
    pred = model.get_prediction(start=0, end=h)
    var = float(interp1d(
        x, pred.forecast_variance, kind='linear', fill_value="extrapolate"
    )(target_date))
    return stats.norm.ppf((1 + float(confidence_level)) / 2) * np.sqrt(var)


def xforecast_ets_stat(
        values, timeline, statistic_type, seasonality=1,
        data_completion=1, aggregation=0
):
    model, x, y, step = _train_ets_model(
        values, timeline, seasonality, data_completion, aggregation
    )
    statistic_type = int(statistic_type)
    if statistic_type == 1:
        return model.alpha
    if statistic_type == 2:
        return model.beta
    if statistic_type == 3:
        return model.gamma
    if statistic_type == 8:
        return step
    y_true = model.model.endog
    y_fitted = model.fittedvalues
    if statistic_type == 4:
        num = np.mean(np.abs(y_true - y_fitted))
        if np.isclose(num, 0, atol=1.e-7):
            return 0
        m = model.model.seasonal_periods or 1
        denom = np.mean(np.abs(y_true[m:] - y_true[:-m]))
        return num / denom
    if statistic_type == 5:
        denom = np.abs(y_true) + np.abs(y_fitted)
        num = 2.0 * np.abs(y_true - y_fitted)
        return np.mean(np.where(np.isclose(num, 0), 0, num / denom))
    if statistic_type == 6:
        return np.mean(np.abs(y_true - y_fitted))
    if statistic_type == 7:
        return np.sqrt(np.mean((y_true - y_fitted) ** 2))
    return Error.errors['#NUM!']


def xforecast_ets_seasonality(
        values, timeline, data_completion=1, aggregation=0
):
    model = _train_ets_model(
        values, timeline, 1, data_completion, aggregation
    )[0]
    seas = model.model.seasonal_periods or 1
    return 0 if seas == 1 else seas


def xforecast(x, a=None, b=None):
    return a + b * x


FUNCTIONS['_XLFN.FORECAST.LINEAR'] = FUNCTIONS['FORECAST'] = wrap_ufunc(
    xforecast, args_parser=_args_parser_forecast, excluded={1, 2},
    input_parser=lambda x, a, b: (_convert_args(x), a, b)
)
FUNCTIONS['FORECAST.LINEAR'] = FUNCTIONS['FORECAST']
FUNCTIONS['_XLFN.FORECAST.ETS.STAT'] = FUNCTIONS[
    'FORECAST.ETS.STAT'
] = wrap_ufunc(
    xforecast_ets_stat, excluded={0, 1},
    check_error=lambda
        values, timeline, statistic_type, seasonality=1,
        data_completion=1, aggregation=0: get_error(
        statistic_type, seasonality, data_completion,
        aggregation
    ),
    args_parser=lambda
        values, timeline, statistic_type, seasonality=1, data_completion=1,
        aggregation=0: (
        replace_empty(values, ""),
        replace_empty(timeline, ""),
        replace_empty(statistic_type),
        replace_empty(seasonality),
        replace_empty(data_completion, 1),
        replace_empty(aggregation)
    ),
    input_parser=lambda
        values, timeline, statistic_type, seasonality=1, data_completion=1,
        aggregation=0: (
        values, timeline, _convert_args(statistic_type),
        _convert_args(seasonality), _convert_args(data_completion),
        _convert_args(aggregation)
    )
)
FUNCTIONS['_XLFN.FORECAST.ETS.SEASONALITY'] = FUNCTIONS[
    'FORECAST.ETS.SEASONALITY'
] = wrap_ufunc(
    xforecast_ets_seasonality, excluded={0, 1},
    check_error=lambda
        values, timeline, data_completion=1, aggregation=0: get_error(
        data_completion, aggregation
    ),
    args_parser=lambda
        values, timeline, data_completion=1, aggregation=0: (
        replace_empty(values, ""),
        replace_empty(timeline, ""),
        replace_empty(data_completion, 1),
        replace_empty(aggregation)
    ),
    input_parser=lambda
        values, timeline, data_completion=1, aggregation=0: (
        values, timeline,
        _convert_args(data_completion), _convert_args(aggregation)
    )
)
FUNCTIONS['_XLFN.FORECAST.ETS.CONFINT'] = FUNCTIONS[
    'FORECAST.ETS.CONFINT'
] = wrap_ufunc(
    xforecast_ets_confint, excluded={1, 2},
    check_error=lambda
        target_date, values, timeline, confidence_level=.95, seasonality=1,
        data_completion=1, aggregation=0: get_error(
        target_date, confidence_level, seasonality, data_completion,
        aggregation
    ),
    args_parser=lambda
        target_date, values, timeline, confidence_level=.95, seasonality=1,
        data_completion=1, aggregation=0: (
        replace_empty(target_date),
        replace_empty(values, ""),
        replace_empty(timeline, ""),
        replace_empty(confidence_level),
        replace_empty(seasonality),
        replace_empty(data_completion, 1),
        replace_empty(aggregation)
    ),
    input_parser=lambda
        target_date, values, timeline, confidence_level=.95, seasonality=1,
        data_completion=1, aggregation=0: (
        _convert_args(target_date), values, timeline,
        _convert_args(confidence_level), _convert_args(seasonality),
        _convert_args(data_completion), _convert_args(aggregation)
    )
)
FUNCTIONS['_XLFN.FORECAST.ETS'] = FUNCTIONS['FORECAST.ETS'] = wrap_ufunc(
    xforecast_ets, excluded={1, 2},
    check_error=lambda
        target_date, values, timeline, seasonality=1,
        data_completion=1, aggregation=0: get_error(
        target_date, seasonality, data_completion, aggregation
    ),
    args_parser=lambda
        target_date, values, timeline, seasonality=1, data_completion=1,
        aggregation=0: (
        replace_empty(target_date),
        replace_empty(values, ""),
        replace_empty(timeline, ""),
        replace_empty(seasonality),
        replace_empty(data_completion, 1),
        replace_empty(aggregation)
    ),
    input_parser=lambda
        target_date, values, timeline, seasonality=1, data_completion=1,
        aggregation=0: (
        _convert_args(target_date), values, timeline,
        _convert_args(seasonality), _convert_args(data_completion),
        _convert_args(aggregation)
    )
)


def _parse_cumulative(cumulative):
    if isinstance(cumulative, str):
        if cumulative.lower() in ('true', 'false'):
            cumulative = cumulative.lower() == 'true'
        else:
            raise FoundError(err=Error.errors['#VALUE!'])
    return cumulative


def xnormdist(z, mu, sigma, cumulative=True):
    if sigma <= 0:
        return Error.errors['#NUM!']
    norm = NormalDist(mu=mu, sigma=sigma)
    return norm.cdf(z) if cumulative else norm.pdf(z)


def xnorminv(z, mu=0, sigma=1):
    if z <= 0.0 or z >= 1.0 or sigma <= 0:
        return Error.errors['#NUM!']
    norm = NormalDist(mu=mu, sigma=sigma)
    return norm.inv_cdf(z)


FUNCTIONS['_XLFN.NORM.DIST'] = FUNCTIONS['NORM.DIST'] = wrap_ufunc(
    xnormdist,
    input_parser=lambda *a: tuple(
        map(_convert_args, a[:-1])
    ) + tuple(map(_parse_cumulative, a[-1:]))
)
FUNCTIONS['_XLFN.NORM.INV'] = FUNCTIONS['NORM.INV'] = wrap_ufunc(
    xnorminv,
    input_parser=lambda *a: tuple(map(_convert_args, a))
)
FUNCTIONS['_XLFN.NORM.S.DIST'] = FUNCTIONS['NORM.S.DIST'] = wrap_ufunc(
    xnormdist,
    input_parser=lambda x, a=1: (_convert_args(x), 0, 1, _parse_cumulative(a))
)
FUNCTIONS['_XLFN.NORM.S.INV'] = FUNCTIONS['NORM.S.INV'] = wrap_ufunc(
    xnorminv,
    input_parser=lambda x: (_convert_args(x),)
)


def xweibulldist(x, alfa, beta, cumulative=True):
    if alfa <= 0 or beta <= 0 or x < 0:
        return Error.errors['#NUM!']

    rv = stats.weibull_min(c=alfa, loc=0.0, scale=beta)
    return float(rv.cdf(x) if cumulative else rv.pdf(x))


FUNCTIONS['_XLFN.WEIBULL.DIST'] = FUNCTIONS['WEIBULL.DIST'] = wrap_ufunc(
    xweibulldist,
    input_parser=lambda *a: tuple(
        map(_convert_args, a[:-1])
    ) + tuple(map(_parse_cumulative, a[-1:]))
)


def xlognormdist(z, mu, sigma, cumulative=True):
    if z <= 0 or sigma <= 0:
        return Error.errors['#NUM!']
    func = stats.lognorm.cdf if cumulative else stats.lognorm.pdf
    return func(z, s=sigma, scale=np.exp(mu))


def xlognorminv(z, mu=0, sigma=1):
    if z <= 0.0 or z > 1.0 or sigma <= 0:
        return Error.errors['#NUM!']
    return stats.lognorm.ppf(z, s=sigma, scale=np.exp(mu))


FUNCTIONS['_XLFN.LOGNORM.DIST'] = FUNCTIONS['LOGNORM.DIST'] = wrap_ufunc(
    xlognormdist,
    input_parser=lambda *a: tuple(
        map(_convert_args, a[:-1])
    ) + tuple(map(_parse_cumulative, a[-1:]))
)
FUNCTIONS['_XLFN.LOGNORM.INV'] = FUNCTIONS['LOGNORM.INV'] = wrap_ufunc(
    xlognorminv,
    input_parser=lambda *a: tuple(map(_convert_args, a))
)


def xbetadist(x, _alpha, _beta, cumulative=True, lb=0, ub=1):
    if x < lb or x > ub or lb >= ub or _alpha <= 0 or _beta <= 0:
        return Error.errors['#NUM!']

    func = stats.beta.cdf if cumulative else stats.beta.pdf
    return func(x, _alpha, _beta, loc=lb, scale=(ub - lb))


def xbetainv(x, _alpha, _beta, lb=0, ub=1):
    if x <= 0.0 or x >= 1.0 or _alpha <= 0 or _beta <= 0 or lb >= ub:
        return Error.errors['#NUM!']
    return stats.beta.ppf(x, _alpha, _beta, loc=lb, scale=(ub - lb))


FUNCTIONS['_XLFN.BETA.DIST'] = FUNCTIONS['BETA.DIST'] = wrap_ufunc(
    xbetadist,
    input_parser=lambda *a: tuple(
        map(_convert_args, a[:3])
    ) + tuple(map(_parse_cumulative, a[3:4])) + tuple(map(_convert_args, a[4:]))
)
FUNCTIONS['_XLFN.BETA.INV'] = FUNCTIONS['BETA.INV'] = wrap_ufunc(
    xbetainv,
    input_parser=lambda *a: tuple(map(_convert_args, a))
)


def xnegbinomdist(number_f, number_s, probability_s, cumulative=True):
    number_f = int(number_f)
    number_s = int(number_s)
    if not (0 <= probability_s <= 1 and number_f >= 0 and number_s >= 1):
        return Error.errors['#NUM!']

    func = stats.nbinom.cdf if cumulative else stats.nbinom.pmf
    return func(number_f, number_s, probability_s)


def xbinomdist(number_s, trials, probability_s, cumulative=True):
    trials = int(trials)
    number_s = int(number_s)
    if not (0 <= probability_s <= 1 and 0 <= number_s <= trials):
        return Error.errors['#NUM!']

    func = stats.binom.cdf if cumulative else stats.binom.pmf
    return func(number_s, trials, probability_s)


def xbinomdistrange(trials, probability_s, number_s, number_s2=None):
    number_s2 = number_s if number_s2 is None else number_s2

    trials = int(trials)
    number_s = int(number_s)
    number_s2 = int(number_s2)
    if not (0 <= probability_s <= 1 and (
            0 <= number_s <= trials and 0 <= number_s2 <= trials
    )):
        return Error.errors['#NUM!']

    func = stats.binom.cdf
    return func(number_s2, trials, probability_s) - func(
        number_s - 1, trials, probability_s
    )


def xbinominv(trials, probability_s, alpha):
    trials = int(trials)
    if not (0 <= probability_s <= 1 and 0 < alpha < 1 and trials >= 0):
        return Error.errors['#NUM!']
    return stats.binom.ppf(alpha, trials, probability_s)


FUNCTIONS['_XLFN.NEGBINOM.DIST'] = FUNCTIONS['NEGBINOM.DIST'] = wrap_ufunc(
    xnegbinomdist,
    input_parser=lambda *a: tuple(
        map(_convert_args, a[:-1])
    ) + tuple(map(_parse_cumulative, a[-1:]))
)
FUNCTIONS['_XLFN.BINOM.DIST'] = FUNCTIONS['BINOM.DIST'] = wrap_ufunc(
    xbinomdist,
    input_parser=lambda *a: tuple(
        map(_convert_args, a[:-1])
    ) + tuple(map(_parse_cumulative, a[-1:]))
)
FUNCTIONS['BINOM.DIST.RANGE'] = wrap_ufunc(
    xbinomdistrange,
    input_parser=lambda *a: tuple(map(_convert_args, a))
)
FUNCTIONS['_XLFN.BINOM.DIST.RANGE'] = FUNCTIONS['BINOM.DIST.RANGE']
FUNCTIONS['_XLFN.BINOM.INV'] = FUNCTIONS['BINOM.INV'] = wrap_ufunc(
    xbinominv,
    input_parser=lambda *a: tuple(map(_convert_args, a))
)


def xchisqdist(x, deg_freedom, cumulative=True):
    deg_freedom = int(deg_freedom)
    if not (x >= 0 and 1 <= deg_freedom <= 1e10):
        return Error.errors['#NUM!']

    func = stats.chi2.cdf if cumulative else stats.chi2.pdf
    return func(x, deg_freedom)


def xchisqdistrt(x, deg_freedom):
    deg_freedom = int(deg_freedom)
    if not (x >= 0 and 1 <= deg_freedom <= 1e10):
        return Error.errors['#NUM!']
    return stats.chi2.sf(x, deg_freedom)


def xchisqinv(probability, deg_freedom):
    deg_freedom = int(deg_freedom)
    if not (0 <= probability <= 1 and 1 <= deg_freedom <= 1e10):
        return Error.errors['#NUM!']
    return stats.chi2.ppf(probability, deg_freedom)


def xchisqinvrt(probability, deg_freedom):
    deg_freedom = int(deg_freedom)
    if not (0 <= probability <= 1 and 1 <= deg_freedom <= 1e10):
        return Error.errors['#NUM!']
    return stats.chi2.isf(probability, deg_freedom)


def _parse_ranges(arr1, arr2, error_x_row=False, raise_diff_len=True):
    arr1 = tuple(flatten(arr1, None))
    arr2 = tuple(flatten(arr2, None))
    if raise_diff_len and len(arr1) != len(arr2):
        raise FoundError(err=Error.errors['#N/A'])
    if not error_x_row:
        err = get_error(arr1, arr2)
        if err:
            if err is Error.errors['#NULL!']:
                err = Error.errors['#NUM!']
            raise FoundError(err=err)
    _arr1 = []
    _arr2 = []
    if raise_diff_len:
        for a, e in zip(arr1, arr2):
            a = a.item() if isinstance(a, np.ndarray) else a
            e = e.item() if isinstance(e, np.ndarray) else e
            if error_x_row:
                err = get_error(a, e)
                if err:
                    if err is Error.errors['#NULL!']:
                        err = Error.errors['#NUM!']
                    raise FoundError(err=err)
            if isinstance(a, bool) or isinstance(e, bool):
                continue
            if isinstance(a, (int, float)) and isinstance(e, (int, float)):
                _arr1.append(a)
                _arr2.append(e)
    else:
        for arr, _arr in ((arr1, _arr1), (arr2, _arr2)):
            for v in arr:
                v = v.item() if isinstance(v, np.ndarray) else v
                if isinstance(v, bool):
                    continue
                if isinstance(v, (int, float)):
                    _arr.append(v)
    return _arr1, _arr2


def xchisqtest(actual_range, expected_range):
    actual_range = np.atleast_2d(actual_range)
    r, c = actual_range.shape
    if r > 1 and c > 1:
        ddof = (r - 1) * (c - 1)
    elif r == 1 and c > 1:
        ddof = c - 1
    elif r > 1 and c == 1:
        ddof = r - 1
    else:
        return Error.errors['#N/A']

    _actual_range, _expected_range = _parse_ranges(actual_range, expected_range)

    if not _actual_range:
        return Error.errors['#DIV/0!']
    test = stats.chisquare(_actual_range, _expected_range, sum_check=False)
    return stats.chi2.sf(test.statistic, ddof)


FUNCTIONS['_XLFN.CHISQ.DIST'] = FUNCTIONS['CHISQ.DIST'] = wrap_ufunc(
    xchisqdist,
    input_parser=lambda *a: tuple(
        map(_convert_args, a[:-1])
    ) + tuple(map(_parse_cumulative, a[-1:]))
)
FUNCTIONS['_XLFN.CHISQ.INV'] = FUNCTIONS['CHISQ.INV'] = wrap_ufunc(
    xchisqinv,
    input_parser=lambda *a: tuple(map(_convert_args, a))
)
FUNCTIONS['_XLFN.CHISQ.DIST.RT'] = FUNCTIONS['CHISQ.DIST.RT'] = wrap_ufunc(
    xchisqdistrt,
    input_parser=lambda *a: tuple(map(_convert_args, a))
)
FUNCTIONS['_XLFN.CHISQ.INV.RT'] = FUNCTIONS['CHISQ.INV.RT'] = wrap_ufunc(
    xchisqinvrt,
    input_parser=lambda *a: tuple(map(_convert_args, a))
)
FUNCTIONS['_XLFN.CHISQ.TEST'] = FUNCTIONS['CHISQ.TEST'] = wrap_func(
    xchisqtest
)


def xconfidence_norm(alpha, standard_dev, size):
    """
    CONFIDENCE.NORM(alpha, standard_dev, size)
    - alpha in (0,1)
    - standard_dev > 0
    - size > 0 (integer)
    Returns the margin of error for a population mean when σ is known.
    """
    n = int(size)
    if not (0 < alpha < 1) or standard_dev <= 0 or n <= 0:
        return Error.errors['#NUM!']
    z = NormalDist().inv_cdf(1 - alpha / 2.0)
    return z * standard_dev / math.sqrt(n)


def xconfidence_t(alpha, standard_dev, size):
    """
    CONFIDENCE.T(alpha, standard_dev, size)
    Returns the margin of error: t_{1-α/2, n-1} * sd / sqrt(n)
    """
    n = int(size)
    if not (0 < alpha < 1) or standard_dev <= 0 or n < 1:
        return Error.errors['#NUM!']
    if n == 1:
        return Error.errors['#DIV/0!']

    tcrit = float(stats.t.ppf(1.0 - alpha / 2.0, n - 1))
    return tcrit * standard_dev / math.sqrt(n)


FUNCTIONS['_XLFN.CONFIDENCE.NORM'] = FUNCTIONS['CONFIDENCE.NORM'] = wrap_ufunc(
    xconfidence_norm,
    input_parser=lambda *a: tuple(map(_convert_args, a))
)
FUNCTIONS['_XLFN.CONFIDENCE.T'] = FUNCTIONS['CONFIDENCE.T'] = wrap_ufunc(
    xconfidence_t,
    input_parser=lambda *a: tuple(map(_convert_args, a))
)


def _cov_core(arr1, arr2, sample: bool):
    """
    Excel-like covariance:
      - sample=False -> COVARIANCE.P  (divide by N)
      - sample=True  -> COVARIANCE.S  (divide by N-1)
    Uses _parse_yxp to (a) align lengths, (b) drop pairs with text/bools,
    and (c) raise Excel-style errors.
    """

    y, x = _parse_ranges(arr1, arr2, True)

    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = y.size
    if sample:
        if n <= 1:
            return Error.errors['#DIV/0!']
        denom = n - 1
    else:
        if n == 0:
            return Error.errors['#DIV/0!']
        denom = n

    ybar = float(y.mean())
    xbar = float(x.mean())
    cov = float(np.dot(y - ybar, x - xbar) / denom)
    return cov


def xcovariance_p(arr1, arr2):
    return _cov_core(arr1, arr2, sample=False)


def xcovariance_s(arr1, arr2):
    return _cov_core(arr1, arr2, sample=True)


FUNCTIONS['_XLFN.COVARIANCE.P'] = FUNCTIONS['COVARIANCE.P'] = wrap_func(
    xcovariance_p
)
FUNCTIONS['_XLFN.COVARIANCE.S'] = FUNCTIONS['COVARIANCE.S'] = wrap_func(
    xcovariance_s
)


def xfdist(x, deg_freedom1, deg_freedom2, cumulative=True):
    deg_freedom1 = int(deg_freedom1)
    deg_freedom2 = int(deg_freedom2)
    if not (x >= 0 and deg_freedom1 >= 1 and deg_freedom2 >= 1):
        return Error.errors['#NUM!']

    func = stats.f.cdf if cumulative else stats.f.pdf
    return func(x, deg_freedom1, deg_freedom2)


def xfdistrt(x, deg_freedom1, deg_freedom2):
    deg_freedom1 = int(deg_freedom1)
    deg_freedom2 = int(deg_freedom2)
    if not (x >= 0 and deg_freedom1 >= 1 and deg_freedom2 >= 1):
        return Error.errors['#NUM!']
    return stats.f.sf(x, deg_freedom1, deg_freedom2)


def xfinv(probability, deg_freedom1, deg_freedom2):
    deg_freedom1 = int(deg_freedom1)
    deg_freedom2 = int(deg_freedom2)
    if not (0 <= probability <= 1 and deg_freedom1 >= 1 and deg_freedom2 >= 1):
        return Error.errors['#NUM!']
    return stats.f.ppf(probability, deg_freedom1, deg_freedom2)


def xfinvrt(probability, deg_freedom1, deg_freedom2):
    deg_freedom1 = int(deg_freedom1)
    deg_freedom2 = int(deg_freedom2)
    if not (0 <= probability <= 1 and deg_freedom1 >= 1 and deg_freedom2 >= 1):
        return Error.errors['#NUM!']
    return stats.f.isf(probability, deg_freedom1, deg_freedom2)


def xftest(array1, array2):
    _array1, _array2 = _parse_ranges(array1, array2, raise_diff_len=False)
    _array1, _array2 = np.asarray(_array1, float), np.asarray(_array2, float)
    if _array1.size < 2 or _array2.size < 2:
        return Error.errors['#DIV/0!']

    va = float(np.var(_array1, ddof=1))
    vb = float(np.var(_array2, ddof=1))
    if np.isclose(va, 0) or np.isclose(vb, 0):
        return Error.errors['#DIV/0!']

    # Put larger variance on top so F >= 1
    if va >= vb:
        F = va / vb
        d1 = _array1.size - 1
        d2 = _array2.size - 1
    else:
        F = vb / va
        d1 = _array2.size - 1
        d2 = _array1.size - 1

    # One-tail (right) then two-tail
    p_right = float(stats.f.sf(F, d1, d2))
    p_two = min(1.0, 2.0 * p_right)
    return p_two


FUNCTIONS['_XLFN.F.DIST'] = FUNCTIONS['F.DIST'] = wrap_ufunc(
    xfdist,
    input_parser=lambda *a: tuple(
        map(_convert_args, a[:-1])
    ) + tuple(map(_parse_cumulative, a[-1:]))
)
FUNCTIONS['_XLFN.F.INV'] = FUNCTIONS['F.INV'] = wrap_ufunc(
    xfinv,
    input_parser=lambda *a: tuple(map(_convert_args, a))
)
FUNCTIONS['_XLFN.F.DIST.RT'] = FUNCTIONS['F.DIST.RT'] = wrap_ufunc(
    xfdistrt,
    input_parser=lambda *a: tuple(map(_convert_args, a))
)
FUNCTIONS['_XLFN.F.INV.RT'] = FUNCTIONS['F.INV.RT'] = wrap_ufunc(
    xfinvrt,
    input_parser=lambda *a: tuple(map(_convert_args, a))
)
FUNCTIONS['_XLFN.F.TEST'] = FUNCTIONS['F.TEST'] = wrap_func(
    xftest
)


def xt_dist(x, deg_freedom, cumulative=True):
    deg_freedom = int(deg_freedom)
    if deg_freedom <= 0:
        if not cumulative and deg_freedom == 0:
            return Error.errors['#DIV/0!']
        return Error.errors['#NUM!']
    func = stats.t.cdf if cumulative else stats.t.pdf
    return func(x, deg_freedom)


def xt_distrt(x, deg_freedom):
    deg_freedom = int(deg_freedom)
    if not (1 <= deg_freedom):
        return Error.errors['#NUM!']
    return stats.t.sf(x, deg_freedom)


def xt_dist2t(x, deg_freedom):
    deg_freedom = int(deg_freedom)
    if not (x >= 0 and 1 <= deg_freedom):
        return Error.errors['#NUM!']
    p_right = stats.t.sf(x, deg_freedom)
    return min(1.0, 2.0 * p_right)


def xt_inv(probability, deg_freedom):
    deg_freedom = int(deg_freedom)
    if not (0 <= probability <= 1 and 1 <= deg_freedom):
        return Error.errors['#NUM!']
    return stats.t.ppf(probability, deg_freedom)


def xt_inv2t(probability, deg_freedom):
    deg_freedom = int(deg_freedom)
    if not (0 <= probability <= 1 and 1 <= deg_freedom):
        return Error.errors['#NUM!']
    return stats.t.ppf(1.0 - probability / 2., deg_freedom)


def xt_test(array1, array2, tails, ttype):
    if tails not in (1, 2) or ttype not in (1, 2, 3):
        return Error.errors['#NUM!']

    a, b = _parse_ranges(
        array1, array2, raise_diff_len=ttype == 1, error_x_row=ttype == 1
    )
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    # paired
    if ttype == 1:
        if a.size < 2 or b.size < 2:
            return Error.errors['#DIV/0!']
        d = a - b
        n = d.size
        mean = float(d.mean())
        sd = float(d.std(ddof=1))
        if np.isclose(sd, 0):
            return Error.errors['#DIV/0!']
        t = mean / (sd / math.sqrt(n))
        df = n - 1

    # equal variance (pooled)
    elif ttype == 2:
        if a.size < 2 or b.size < 2:
            return Error.errors['#DIV/0!']
        na, nb = a.size, b.size
        ma, mb = float(a.mean()), float(b.mean())
        va, vb = float(a.var(ddof=1)), float(b.var(ddof=1))
        if np.isclose(va, 0) and np.isclose(vb, 0):
            return Error.errors['#DIV/0!']
        sp = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
        if np.isclose(sp, 0):
            return Error.errors['#DIV/0!']
        t = (ma - mb) / (sp * math.sqrt(1.0 / na + 1.0 / nb))
        df = na + nb - 2

    # unequal variance (Welch)
    else:  # ttype == 3
        if a.size < 2 or b.size < 2:
            return Error.errors['#DIV/0!']
        na, nb = a.size, b.size
        ma, mb = float(a.mean()), float(b.mean())
        va, vb = float(a.var(ddof=1)), float(b.var(ddof=1))
        denom = va / na + vb / nb
        if np.isclose(denom, 0):
            return Error.errors['#DIV/0!']
        t = (ma - mb) / math.sqrt(denom)
        # Welch–Satterthwaite df
        df = (denom ** 2) / ((va * va) / ((na * na) * (na - 1)) + (vb * vb) / (
                (nb * nb) * (nb - 1)))
        if not math.isfinite(df) or df <= 0:
            return Error.errors['#DIV/0!']

    # p-values from |t|
    p_one = stats.t.sf(abs(t), df)
    p_two = min(1.0, 2.0 * p_one)
    return p_one if tails == 1 else p_two


FUNCTIONS['_XLFN.T.DIST'] = FUNCTIONS['T.DIST'] = wrap_ufunc(
    xt_dist,
    input_parser=lambda *a: tuple(
        map(_convert_args, a[:-1])
    ) + tuple(map(_parse_cumulative, a[-1:]))
)
FUNCTIONS['_XLFN.T.INV'] = FUNCTIONS['T.INV'] = wrap_ufunc(
    xt_inv,
    input_parser=lambda *a: tuple(map(_convert_args, a))
)
FUNCTIONS['_XLFN.T.DIST.2T'] = FUNCTIONS['T.DIST.2T'] = wrap_ufunc(
    xt_dist2t,
    input_parser=lambda *a: tuple(map(_convert_args, a))
)
FUNCTIONS['_XLFN.T.DIST.RT'] = FUNCTIONS['T.DIST.RT'] = wrap_ufunc(
    xt_distrt,
    input_parser=lambda *a: tuple(map(_convert_args, a))
)
FUNCTIONS['_XLFN.T.INV.2T'] = FUNCTIONS['T.INV.2T'] = wrap_ufunc(
    xt_inv2t,
    input_parser=lambda *a: tuple(map(_convert_args, a))
)
FUNCTIONS['_XLFN.T.TEST'] = FUNCTIONS['T.TEST'] = wrap_ufunc(
    xt_test,
    input_parser=lambda a1, a2, *a: (a1, a2) + tuple(map(_convert_args, a)),
    check_error=lambda *a: None, args_parser=lambda *a: a, excluded={0, 1}
)


def xexpon_dist(x, rate, cumulative):
    """
    EXPON.DIST(x, lambda, cumulative)
      - cumulative TRUE  -> 1 - exp(-lambda * x)
      - cumulative FALSE -> lambda * exp(-lambda * x)
      - Excel errors:
          * x < 0 or lambda <= 0 -> #NUM!
          * cumulative not TRUE/FALSE -> #VALUE!
    """

    if x < 0 or rate <= 0:
        return Error.errors['#NUM!']

    if cumulative:
        return 1.0 - np.exp(-rate * x)
    else:
        return rate * np.exp(-rate * x)


FUNCTIONS['_XLFN.EXPON.DIST'] = FUNCTIONS['EXPON.DIST'] = wrap_ufunc(
    xexpon_dist,
    input_parser=lambda *a: tuple(
        map(_convert_args, a[:-1])
    ) + tuple(map(_parse_cumulative, a[-1:]))
)


def xpoisson_dist(x, mean, cumulative):
    if x < 0 or mean < 0:
        return Error.errors['#NUM!']
    func = stats.poisson.cdf if cumulative else stats.poisson.pmf
    return func(x, mean)


FUNCTIONS['_XLFN.POISSON.DIST'] = FUNCTIONS['POISSON.DIST'] = wrap_ufunc(
    xpoisson_dist,
    input_parser=lambda *a: tuple(
        map(_convert_args, a[:-1])
    ) + tuple(map(_parse_cumulative, a[-1:]))
)


def xfisher(number):
    """
    FISHER(number) -> 0.5 * ln((1+x)/(1-x))
    Excel domain: -1 < x < 1  (else #NUM!)
    """
    x = number
    if not (-1.0 < x < 1.0):
        return Error.errors['#NUM!']
    return 0.5 * np.log((1.0 + x) / (1.0 - x))


def xfisherinv(y):
    """
    FISHERINV(y) -> inverse Fisher transform = tanh(y)
    - Domain: any real y (returns value in (-1, 1)).
    """
    return np.tanh(y)


FUNCTIONS['FISHER'] = wrap_ufunc(
    xfisher,
    input_parser=lambda x: (_convert_args(x),)
)
FUNCTIONS['FISHERINV'] = wrap_ufunc(
    xfisherinv,
    input_parser=lambda y: (_convert_args(y),)
)


def xphi(y):
    return NormalDist().pdf(y)


FUNCTIONS['_XLFN.PHI'] = FUNCTIONS['PHI'] = wrap_ufunc(
    xphi, input_parser=lambda x: (_convert_args(x),)
)


def xgamma(number):
    """
    GAMMA(number)
    Excel domain:
      - allowed: any real except non-positive integers
      - error:   number in { …, -2, -1, 0 }  -> #NUM!
    """
    if number <= 0 and float(number).is_integer():
        return Error.errors['#NUM!']
    try:
        return float(math.gamma(number))
    except (ValueError, OverflowError):
        return Error.errors['#NUM!']


def xgamma_dist(x, alpha, beta, cumulative=True):
    if x < 0 or alpha <= 0 or beta <= 0:
        return Error.errors['#NUM!']
    rv = stats.gamma(alpha, loc=0.0, scale=beta)
    return rv.cdf(x) if cumulative else rv.pdf(x)


def xgamma_inv(probability, alpha, beta):
    """
    GAMMA.INV(probability, alpha, beta)
    - 0 < probability < 1
    - alpha > 0, beta > 0
    """
    if not (0.0 <= probability <= 1.0) or alpha <= 0 or beta <= 0:
        return Error.errors['#NUM!']
    return stats.gamma.ppf(probability, alpha, loc=0.0, scale=beta)


def xgammaln(x):
    """
    GAMMALN(x) and GAMMALN.PRECISE(x)
    Excel domain: x > 0  (else #NUM!)
    """
    if x <= 0:
        return Error.errors['#NUM!']
    return math.lgamma(x)


FUNCTIONS['_XLFN.GAMMA'] = FUNCTIONS['GAMMA'] = wrap_ufunc(
    xgamma,
    input_parser=lambda x: (_convert_args(x),)
)
FUNCTIONS['_XLFN.GAMMA.DIST'] = FUNCTIONS['GAMMA.DIST'] = wrap_ufunc(
    xgamma_dist,
    input_parser=lambda *a: tuple(
        map(_convert_args, a[:-1])
    ) + tuple(map(_parse_cumulative, a[-1:]))
)
FUNCTIONS['_XLFN.GAMMA.INV'] = FUNCTIONS['GAMMA.INV'] = wrap_ufunc(
    xgamma_inv,
    input_parser=lambda *a: tuple(map(_convert_args, a))
)
FUNCTIONS['_XLFN.GAMMALN.PRECISE'] = FUNCTIONS['GAMMALN.PRECISE'] = wrap_ufunc(
    xgammaln, input_parser=lambda x: (_convert_args(x),)
)
FUNCTIONS['_XLFN.GAMMALN'] = FUNCTIONS['GAMMALN'] = FUNCTIONS['GAMMALN.PRECISE']


def xgauss(x):
    """
    GAUSS(z) = Φ(z) - 0.5
    (Φ is the CDF of N(0,1); result in (-0.5, 0.5))
    """
    return NormalDist().cdf(x) - 0.5


FUNCTIONS['_XLFN.GAUSS'] = FUNCTIONS['GAUSS'] = wrap_ufunc(
    xgauss,
    input_parser=lambda x: (_convert_args(x),)
)


def xhypergeom_dist(
        sample_s, number_sample, population_s, number_pop, cumulative):
    """
    EXPON.DIST(x, lambda, cumulative)
      - cumulative TRUE  -> 1 - exp(-lambda * x)
      - cumulative FALSE -> lambda * exp(-lambda * x)
      - Excel errors:
          * x < 0 or lambda <= 0 -> #NUM!
          * cumulative not TRUE/FALSE -> #VALUE!
    """

    k = int(sample_s)
    n = int(number_sample)
    K = int(population_s) or 1
    M = int(number_pop) or 1
    # Excel-like domain checks
    if M < 0 or n < 0 or K < 0 or K > M or n > M or k < 0:
        return Error.errors['#NUM!']

    rv = stats.hypergeom(M, K, n)
    return rv.cdf(k) if cumulative else rv.pmf(k)


FUNCTIONS['_XLFN.HYPGEOM.DIST'] = FUNCTIONS['HYPGEOM.DIST'] = wrap_ufunc(
    xhypergeom_dist,
    input_parser=lambda *a: tuple(
        map(_convert_args, a[:-1])
    ) + tuple(map(_parse_cumulative, a[-1:]))
)


def xpearson(array1, array2):
    y, x = _parse_ranges(array1, array2, error_x_row=True)
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    if y.size < 2:
        return Error.errors['#DIV/0!']

    dy = y - y.mean()
    dx = x - x.mean()

    syy = float(np.dot(dy, dy))
    sxx = float(np.dot(dx, dx))
    if np.isclose(sxx, 0) or np.isclose(syy, 0):
        return Error.errors['#DIV/0!']

    return float(np.dot(dx, dy) / math.sqrt(sxx * syy))


FUNCTIONS['_XLFN.PEARSON'] = FUNCTIONS['PEARSON'] = wrap_func(xpearson)


def xpercentrank(exc, vals, x, significance=3):
    """
    PERCENTRANK.EXC(array, x, [significance])
    - Returns rank of x in array as a percentage in (0,1) (excludes endpoints).
    - If x is outside [min(array), max(array)] -> #N/A
    - Rounds to 'significance' decimal places (default 3).
    """
    if not vals:
        return Error.errors['#N/A']
    n = len(vals)

    # outside range -> #N/A (exclusive)
    if x < vals[0] or x > vals[-1]:
        return Error.errors['#N/A']

    # default/validate significance (decimal places to round)
    sig = int(significance)
    if sig < 1:
        return Error.errors['#NUM!']

    # Find position (0-based)
    lo = int(np.searchsorted(vals, x, side='left'))
    hi = int(np.searchsorted(vals, x, side='right'))

    if lo == hi:
        # xx strictly between vals[lo-1] and vals[lo] -> interpolate
        y = vals[lo - 1]
        z = vals[lo]
        # y < xx < z guaranteed; z - y > 0
        frac = (x - y) / (z - y)
        r = (lo - 1) + frac
    else:
        # xx equals a tie block -> average the tied ranks (0-based)
        r = 0.5 * (lo + (hi - 1))
    if exc:
        pct = (r + 1) / (n + 1)
    else:
        pct = r / (n - 1)
    return math.trunc(pct * 10 ** sig) / 10 ** sig


_percentrank_kw = {
    'excluded': {0},
    'input_parser': lambda v, q, s=3: (v, _convert_args(q), _convert_args(s)),
    'check_error': lambda *a: get_error(*a[::-1]),
    'args_parser': lambda v, q, s=3: (sorted(_parse_ranges(
        v, [], error_x_row=False, raise_diff_len=False
    )[0]), replace_empty(q), replace_empty(s))
}
FUNCTIONS['_XLFN.PERCENTRANK.EXC'] = FUNCTIONS['PERCENTRANK.EXC'] = wrap_ufunc(
    functools.partial(xpercentrank, True), **_percentrank_kw
)
FUNCTIONS['_XLFN.PERCENTRANK.INC'] = FUNCTIONS['PERCENTRANK.INC'] = wrap_ufunc(
    functools.partial(xpercentrank, False), **_percentrank_kw
)

_percentile_kw = {
    'excluded': {0},
    'input_parser': lambda v, q: (v, _convert_args(q)),
    'check_error': lambda *a: get_error(*a[::-1]),
    'args_parser': lambda v, q: (
        list(flatten(v, drop_empty=True)), replace_empty(q)
    )
}


def xpercentile(v, p, exclusive=False):
    if len(v) == 0 or not is_number(p) or p < 0 or p > 1:
        return Error.errors['#NUM!']
    if exclusive:
        n = len(v)
        rank = (n + 1) * p
        if rank < 1 or rank > n:
            return Error.errors['#NUM!']
    return np.percentile(v, p * 100, method=exclusive and 'weibull' or 'linear')


FUNCTIONS['_XLFN.PERCENTILE.EXC'] = FUNCTIONS['PERCENTILE.EXC'] = wrap_ufunc(
    functools.partial(xpercentile, exclusive=True), **_percentile_kw
)
FUNCTIONS['_XLFN.PERCENTILE.INC'] = FUNCTIONS['PERCENTILE.INC'] = wrap_ufunc(
    xpercentile, **_percentile_kw
)


def xquartile(v, q, exclusive=False):
    if len(v) == 0:
        return Error.errors['#NUM!']
    if exclusive:
        n = len(v)
        rank = (n + 1) * q * 0.25
        if q <= 0 or q >= 4 or rank < 1 or rank > n:
            return Error.errors['#NUM!']
        method = 'weibull'
    else:
        if q < 0 or q > 4:
            return Error.errors['#NUM!']
        method = 'linear'
    return np.quantile(v, q * 0.25, method=method)


_quartile_kw = sh.combine_dicts(_percentile_kw, {
    'excluded': {0},
    'input_parser': lambda v, q: (v, np.floor(_convert_args(q))),
    'check_error': lambda *a: get_error(*a[::-1]),
    'args_parser': lambda v, q: (
        list(flatten(v, drop_empty=True)), replace_empty(q)
    )
})
FUNCTIONS['_XLFN.QUARTILE.EXC'] = FUNCTIONS['QUARTILE.EXC'] = wrap_ufunc(
    functools.partial(xquartile, exclusive=True), **_quartile_kw
)
FUNCTIONS['_XLFN.QUARTILE.INC'] = FUNCTIONS['QUARTILE.INC'] = wrap_ufunc(
    xquartile, **_quartile_kw
)


def xstdev(args, ddof=1, func=np.std):
    if len(args) <= ddof:
        return Error.errors['#DIV/0!']
    return func(args, ddof=ddof)


FUNCTIONS['_XLFN.STDEV.S'] = FUNCTIONS['STDEV.S'] = wrap_func(functools.partial(
    xfunc, func=xstdev
))
FUNCTIONS['_XLFN.STDEV.P'] = FUNCTIONS['STDEV.P'] = wrap_func(functools.partial(
    xfunc, func=functools.partial(xstdev, ddof=0), default=None
))
FUNCTIONS['STDEVA'] = wrap_func(functools.partial(
    xfunc, convert=_convert, check=is_not_empty, func=xstdev
))
FUNCTIONS['STDEVPA'] = wrap_func(functools.partial(
    xfunc, convert=_convert, check=is_not_empty, func=functools.partial(
        xstdev, ddof=0
    ), default=None
))

FUNCTIONS['_XLFN.VAR.S'] = FUNCTIONS['VAR.S'] = wrap_func(functools.partial(
    xfunc, func=functools.partial(xstdev, func=np.var)
))
FUNCTIONS['_XLFN.VAR.P'] = FUNCTIONS['VAR.P'] = wrap_func(functools.partial(
    xfunc, func=functools.partial(xstdev, ddof=0, func=np.var), default=None
))
FUNCTIONS['VARA'] = wrap_func(functools.partial(
    xfunc, convert=_convert, check=is_not_empty, func=functools.partial(
        xstdev, func=np.var
    )
))
FUNCTIONS['VARPA'] = wrap_func(functools.partial(
    xfunc, convert=_convert, check=is_not_empty, func=functools.partial(
        xstdev, ddof=0, func=np.var
    ), default=None
))


def xpermut(number, number_chosen):
    number = int(number)
    number_chosen = int(number_chosen)
    if number_chosen < 0 or number < 0 or number < number_chosen:
        return Error.errors['#NUM!']
    return math.factorial(number) / math.factorial(number - number_chosen)


def xpermutationa(number, number_chosen):
    if number_chosen < 0 or number < 0:
        return Error.errors['#NUM!']
    return int(number) ** int(number_chosen)


FUNCTIONS['_XLFN.PERMUT'] = FUNCTIONS['PERMUT'] = wrap_ufunc(xpermut)
FUNCTIONS['_XLFN.PERMUTATIONA'] = FUNCTIONS['PERMUTATIONA'] = wrap_ufunc(
    xpermutationa
)


def xrank(method, number, ref, order=0):
    arr = np.asarray(ref, float)
    b = arr == float(number)
    if arr.size == 0 or not b.any():
        return Error.errors['#N/A']

    if int(order) == 0:
        arr = -arr

    r = stats.rankdata(arr, method=method)  # 1-based ranks
    return float(r[np.where(b)[0][0]])


# ---- Registration ----
FUNCTIONS['_XLFN.RANK.EQ'] = FUNCTIONS['RANK.EQ'] = wrap_ufunc(
    functools.partial(xrank, "min"),
    excluded={1},
    args_parser=lambda number, ref, order=0: (
        replace_empty(number), ref, replace_empty(order)
    ),
    input_parser=lambda number, ref, order=0: (
        _convert_args(number), sorted(_parse_ranges(
            ref, [], error_x_row=False, raise_diff_len=False
        )[0]), _convert_args(order)
    )
)
FUNCTIONS['_XLFN.RANK.AVG'] = FUNCTIONS['RANK.AVG'] = wrap_ufunc(
    functools.partial(xrank, "average"),
    excluded={1},
    args_parser=lambda number, ref, order=0: (
        replace_empty(number), ref, replace_empty(order)
    ),
    input_parser=lambda number, ref, order=0: (
        _convert_args(number), _parse_ranges(
            ref, [], error_x_row=False,
            raise_diff_len=False
        )[0], _convert_args(order)
    )
)


def xrsq(known_ys, known_xs):
    """
    RSQ(known_y's, known_x's) -> r^2
    """
    known_ys, known_xs = _parse_ranges(known_ys, known_xs, error_x_row=True)

    if len(known_ys) <= 1:
        return Error.errors['#DIV/0!']

    return stats.linregress(known_ys, known_xs)[2] ** 2


def xsteyx(known_ys, known_xs):
    known_ys, known_xs = _parse_ranges(known_ys, known_xs, error_x_row=True)
    known_xs = np.asarray(known_xs, float)
    known_ys = np.asarray(known_ys, float)
    n = known_ys.size
    if n <= 2:
        return Error.errors['#DIV/0!']
    x = known_xs - known_xs.mean()
    y = known_ys - known_ys.mean()
    see = np.dot(y, y) - np.dot(x, y) ** 2 / np.dot(x, x)
    return np.sqrt(max(0.0, see / (n - 2)))


FUNCTIONS['RSQ'] = wrap_func(xrsq)
FUNCTIONS['STEYX'] = wrap_func(xsteyx)


def xstandardize(x, mean, standard_dev):
    if standard_dev <= 0:
        return Error.errors['#NUM!']
    return (x - mean) / standard_dev


FUNCTIONS['STANDARDIZE'] = wrap_ufunc(
    xstandardize, input_parser=lambda *a: tuple(map(_convert_args, a))
)


def xtrimmean(array, percent):
    """
    TRIMMEAN(array, percent)
      - percent in [0,1); trims floor(percent*n/2) from each tail
      - ignores text/booleans/blanks
      - errors:
          * no numeric data                     -> #NUM!
          * percent < 0 or >= 1                 -> #NUM!
          * trims away all data (2k >= n)       -> #NUM!
    """
    # collect finite numerics only
    n = array.size
    k = int(math.floor(percent * n / 2.0))
    if not n or not (0.0 <= percent < 1.0) or 2 * k >= n:
        return Error.errors['#NUM!']

    trimmed = array[k:n - k]
    return float(trimmed.mean())


FUNCTIONS['TRIMMEAN'] = wrap_ufunc(
    xtrimmean,
    excluded={0},
    args_parser=lambda array, percent: (
        np.sort(np.array(_parse_ranges(
            array, [], error_x_row=False,
            raise_diff_len=False
        )[0], dtype=float)), replace_empty(percent)
    ),
    input_parser=lambda array, percent: (array, float(_convert_args(percent)))
)


def xz_test(array, x, sigma=None):
    """
    Z.TEST(array, x, [sigma]) -> one-tailed p-value
      - If sigma is omitted/empty, use sample stdev (ddof=1).
      - Returns #N/A if no numeric data (or n<2 when sigma omitted).
      - Returns #DIV/0! if sigma <= 0 (or sample stdev is 0).
    """
    array = np.array(_parse_ranges(
        array, [], error_x_row=False,
        raise_diff_len=False
    )[0], dtype=float)
    n = array.size
    if n <= 1:
        return Error.errors['#DIV/0!']
    x = float(x)
    if sigma is None:
        sigma = float(np.std(array, ddof=1))
    if sigma <= 0.0:
        return Error.errors['#NUM!']

    return 1 - NormalDist().cdf((np.mean(array) - x) / (sigma / np.sqrt(n)))


FUNCTIONS['_XLFN.Z.TEST'] = FUNCTIONS['Z.TEST'] = wrap_ufunc(
    xz_test,
    excluded={0},
    check_error=lambda array, x, sigma=None: get_error(x) or get_error(
        sigma
    ) or get_error(array),
    args_parser=lambda array, x, sigma=None: (
        array, replace_empty(x), replace_empty(sigma, None)
    ),
    input_parser=lambda array, x, sigma: (
        array, float(_convert_args(x)), _convert_args(sigma)
    )
)


def xfrequency(data_array, bins_array):
    """
    FREQUENCY(data_array, bins_array) -> list of counts
      counts[i] = # of data <= bins_sorted[i] and > previous bin
      counts[-1] = # of data > last bin
    """
    raise_errors(data_array, bins_array)
    x = np.array(_parse_ranges(
        data_array, [], error_x_row=False,
        raise_diff_len=False
    )[0], dtype=float)
    bins = np.array(_parse_ranges(
        bins_array, [], error_x_row=False,
        raise_diff_len=False
    )[0] or [0], dtype=float)

    bins.sort()
    res = np.zeros(bins.size + 1, dtype=int)
    if x.size > 0:
        bins, i, n = np.unique(bins, return_index=True, return_counts=True)
        i = np.append(i, res.size - 1) - np.append([0], n - 1)
        idx = np.searchsorted(bins, x, side='left')
        idx, counts = np.unique_counts(idx)
        res[i[idx]] = counts
    return np.atleast_2d(res).T.view(Array)


FUNCTIONS['FREQUENCY'] = wrap_func(xfrequency)


def xprob(x_range, prob_range, lower_limit, upper_limit=None):
    """
    PROB(x_range, prob_range, lower_limit, [upper_limit])

    Excel-like behavior:
      - Pairwise: ignore pairs where either x or p is text/boolean.
      - Errors:
          * length mismatch (before filtering) -> #N/A (via _parse_yxp)
          * no valid numeric pairs             -> #N/A
          * any probability < 0 or > 1         -> #NUM!
          * sum(probabilities) != 1 (±tol)     -> #NUM!
      - If upper_limit omitted -> P(X == lower_limit).
      - If upper_limit provided -> P(lower_limit <= X <= upper_limit).
      - If lower_limit > upper_limit -> 0.
    """

    xs, ps = _parse_ranges(x_range, prob_range)

    x = np.asarray(xs, dtype=float)
    p = np.asarray(ps, dtype=float)

    if x.size == 0:
        return Error.errors['#N/A']

    if np.any((p < 0.0) | (p > 1.0)) or not np.isclose(p.sum(), 1.0):
        return Error.errors['#NUM!']

    lo = lower_limit
    if upper_limit is None:
        return p[x == lo].sum()

    hi = float(upper_limit)
    if lo > hi:
        return 0.0

    return p[(x >= lo) & (x <= hi)].sum()


FUNCTIONS['PROB'] = wrap_ufunc(
    xprob,
    excluded={0, 1},
    check_nan=False,
    args_parser=lambda x_range, prob_range, lower_limit, upper_limit=None: (
        x_range, prob_range, replace_empty(lower_limit),
        replace_empty(upper_limit)
    ),
    input_parser=lambda x_range, prob_range, lower_limit, upper_limit=None: (
        x_range, prob_range, float(_convert_args(lower_limit)),
        _convert_args(upper_limit)
    )
)


def _parse_linest(known_y, known_x=None, new_x=None):
    assert all(isinstance(v, (float, int)) for v in flatten(known_y, None))
    y = np.asarray(known_y, float)
    if known_x is None:
        y = y.ravel()
        x = np.arange(1, y.size + 1, dtype=float)
    else:
        assert all(isinstance(v, (float, int)) for v in flatten(known_x, None))
        x = np.asarray(known_x, float)
    if x.shape == y.shape:
        x = np.atleast_2d(x.ravel())
        y = np.atleast_2d(y.ravel())
    if new_x is None:
        _new_x = x
    else:
        assert all(isinstance(v, (float, int)) for v in flatten(new_x, None))
        _new_x = np.asarray(new_x, float)
    if 1 not in y.shape or not (
            (y.shape[0] == 1 and y.shape[1] == x.shape[1]) or
            (y.shape[1] == 1 and y.shape[0] == x.shape[0])
    ) or not (
            (y.shape[0] == 1 and x.shape[0] == _new_x.shape[0]) or
            (y.shape[1] == 1 and x.shape[1] == _new_x.shape[1])
    ):
        raise FoundError(err=Error.errors['#REF!'])
    elif y.shape[0] == 1:
        y = y.T
        x = x.T
        _new_x = _new_x.T

    return x, y, _new_x


def _xlinest_stats(x, y, p, const, res):
    yhat = x @ p
    ybar = np.mean(y) if const else 0
    res[4, 0] = ssreg = np.sum((yhat - ybar) ** 2)
    resid = y - yhat
    res[4, 1] = sse = (resid.T @ resid).item()

    sstot = sse + ssreg
    res[2, 0] = r2 = ssreg / sstot
    res[3, 1] = df = x.shape[0] - x.shape[1]
    sigma2 = sse / df
    res[2, 1] = sey = np.sqrt(sigma2)
    XtX_inv = np.linalg.inv(x.T @ x)
    se = np.sqrt(np.maximum(0.0, np.diag(sigma2 * XtX_inv)))
    res[1, :se.size] = se
    res[3, 0] = f = ssreg / sse * df / x.shape[1]


def _xlinest(x, y, const, _stats):
    if const:
        x = np.column_stack([np.ones(y.shape, dtype=float), x])
    p = linalg.lstsq(x, y)[0]
    res = np.empty((5 if _stats else 1, x.shape[1] + int(not const)), object)
    res[:, :] = Error.errors['#N/A']
    if _stats:
        _xlinest_stats(x, y, p, const, res)

    if not const:
        p = np.append([0], p)

    res[0, :] = p[::-1].ravel()
    return res


def _xlinest_parse(known_y, known_x=None, const=True, _stats=False, new_x=None):
    const = _convert_args(next(flatten([const], None)))
    _stats = _convert_args(next(flatten([_stats], None)))
    if get_error(known_y, known_x, new_x, const, _stats):
        raise FoundError(err=Error.errors['#VALUE!'])
    x, y, _new_x = _parse_linest(known_y, known_x, new_x)
    return x, y, const, _stats, _new_x


def xlinest(known_y, known_x=None, const=True, _stats=False):
    x, y, const, _stats = _xlinest_parse(
        known_y, known_x, const, _stats
    )[:-1]
    return _xlinest(x, y, const, _stats).view(Array)


def xlogest(known_y, known_x=None, const=True, _stats=False):
    x, y, const, _stats = _xlinest_parse(
        known_y, known_x, const, _stats
    )[:-1]
    res = _xlinest(x, np.log(y), const, _stats)
    res[0, :] = np.exp(res[0, :].astype(float))
    return res.view(Array)


FUNCTIONS['LINEST'] = wrap_func(xlinest)
FUNCTIONS['LOGEST'] = wrap_func(xlogest)


def xtrend(known_y, known_x=None, new_x=None, const=True):
    x, y, const, _stats, new_x = _xlinest_parse(
        known_y, known_x, const, False, new_x
    )
    if const:
        new_x = np.column_stack([np.ones(new_x.shape[0], dtype=float), new_x])
    res = new_x @ _xlinest(x, y, const, _stats).T[::-1]
    return res.view(Array)


def xgrowth(known_y, known_x=None, new_x=None, const=True):
    x, y, const, _stats, new_x = _xlinest_parse(
        known_y, known_x, const, False, new_x
    )
    if const:
        new_x = np.column_stack([np.ones(new_x.shape[0], dtype=float), new_x])
    res = new_x @ _xlinest(x, np.log(y), const, _stats).T[::-1]
    res = np.exp(res.astype(float))
    return res.view(Array)


FUNCTIONS['TREND'] = wrap_func(xtrend)
FUNCTIONS['GROWTH'] = wrap_func(xgrowth)
