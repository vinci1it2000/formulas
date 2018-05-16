#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
Python equivalents of logical excel functions.
"""
import numpy as np
from . import wrap_func

FUNCTIONS = {}
FUNCTIONS['IF'] = wrap_func(lambda c, x=True, y=False: np.where(c, x, y))


def iferror(val, val_if_error):
    from .info import iserror
    return np.where(iserror(val), val_if_error, val)


FUNCTIONS['IFERROR'] = iferror
