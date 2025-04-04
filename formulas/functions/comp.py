#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2025 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
Python equivalents of compatibility Excel functions.
"""
from .stat import FUNCTIONS as FSTAT

FUNCTIONS = {
    'NORMDIST': FSTAT['NORM.DIST'],
    'NORMINV': FSTAT['NORM.INV'],
    'NORMSDIST': FSTAT['NORM.S.DIST'],
    'NORMSINV': FSTAT['NORM.S.INV'],
    'PERCENTILE': FSTAT['PERCENTILE.INC'],
    'QUARTILE': FSTAT['QUARTILE.INC'],
    'STDEV': FSTAT['STDEV.S'],
    'STDEVP': FSTAT['STDEV.P'],
    'VAR': FSTAT['VAR.S'],
    'VARP': FSTAT['VAR.P']
}
