#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
Defines the formulas exception.
"""


class BaseError(Exception):
    msg = ''

    def __init__(self, *args):
        super(BaseError, self).__init__(self.msg, *args)


class FoundError(BaseError):
    # noinspection PyUnusedLocal
    def __init__(self, *args, err=None, **kwargs):
        self.err = err


class FormulaError(BaseError):
    msg = 'Not a valid formula:\n%s'


class TokenError(FormulaError):
    msg = 'Invalid string: %s'


class ParenthesesError(FormulaError):
    msg = 'Mismatched or misplaced parentheses!'


class FunctionError(FormulaError):
    msg = 'Function not implemented!'


class RangeValueError(BaseError):
    msg = 'Range %s has not value!'


class BroadcastError(BaseError):
    msg = 'Broadcast is not implemented!'
