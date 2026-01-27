#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2025 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
Python equivalents of google Excel functions.
"""
from ..tokens.operand import Error
from . import wrap_func


# noinspection PyUnusedLocal
def xdummy(*args):
    return Error.errors['#NAME?']


FUNCTIONS = {}
FUNCTIONS['__XLUDF.DUMMYFUNCTION'] = FUNCTIONS['DUMMYFUNCTION'] = wrap_func(
    xdummy
)
