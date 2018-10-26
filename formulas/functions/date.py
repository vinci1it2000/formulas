#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
Python equivalents of information excel functions.
"""
import numpy as np
from . import wrap_ranges_func, Error, Array, XlError, wrap_ufunc, wrap_func
import datetime
import calendar

FUNCTIONS = {}

def today():
    return datetime.datetime.now()

FUNCTIONS['TODAY'] = wrap_func(today)

def date(year, month, day):
    dt = datetime.datetime(year=year, month=month, day=day)
    return dt

FUNCTIONS['DATE'] = wrap_func(date)

def _date_attr(input, attr):
    x = input[0,0]
    if isinstance(x, XlError):
        return x
    if(x and hasattr(x, attr)):
        return getattr(x, attr, None)
    return Error.errors['#VALUE!']

def day(date):
    return _date_attr(date, "day")
FUNCTIONS['DAY'] = wrap_func(day)

def year(date):
    return _date_attr(date, "year")

FUNCTIONS['YEAR'] = wrap_func(year)

def month(date):
    return _date_attr(date, "month")
FUNCTIONS['MONTH'] = wrap_func(month)

def yearfrac(date1, date2, basis=0):
    date1 = date1[0,0]
    date2 = date2[0,0]

    if date1 == date2:
        return 0.0
    (y1, m1, d1) = (date1.year, date1.month, date1.day)
    (y2, m2, d2) = (date2.year, date2.month, date2.day)
    eom1 = calendar.monthrange(y1, m1)[1]
    eom2 = calendar.monthrange(y2, m2)[1]
    factor = 0
    if(basis == 0):
        # US 30/360
        if((m1 == m2 == 2) and (eom1 == d1) and (eom2 == d2)):
            #Both dates fall at the end of feb
            d2 = 30
        if(d1 == 31 or (d1 == eom1 and m1 == 2)):
            d1 = 30
        if(d1 == 30 and d2 == 31):
            d2 = 30
        factor = 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)
        factor = factor/360
    elif(basis == 1):
        #Actual/Actual
        #https://github.com/miradulo/isda_daycounters/blob/master/isda_daycounters/actualactual.py
        start_date = datetime.datetime.combine(date1, datetime.datetime.min.time())
        end_date = datetime.datetime.combine(date2, datetime.datetime.min.time())

        year_1_diff = 366 if calendar.isleap(y1) else 365
        year_2_diff = 366 if calendar.isleap(y2) else 365

        total_sum = y2 - y1 - 1
        diff_first = datetime.datetime(y1 + 1, 1, 1) - start_date
        total_sum += diff_first.days / year_1_diff
        diff_second = end_date - datetime.datetime(y2, 1, 1)
        total_sum += diff_second.days / year_2_diff
        factor = total_sum
    elif(basis == 2):
        # Actual/360
        diff = d2 - d1
        factor = diff.days/360
    elif(basis == 3):
        # Actual/365
        diff = d2 - d1
        factor = diff.days/365
    elif(basis == 4):
        # European 30/360
        if(d1 == 31):
            d1 = 30
        if(d2 == 31):
            d2 = 30

        factor = 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)
        factor = factor/360
    return factor


FUNCTIONS['YEARFRAC'] = wrap_func(yearfrac)
