#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
It provides tokens needed to parse the excel formulas.

Sub-Modules:

.. currentmodule:: formulas.tokens

.. autosummary::
    :nosignatures:
    :toctree: tokens/

    ~function
    ~operand
    ~operator
    ~parenthesis
"""
from ..errors import TokenError


class Token(object):
    _re = None

    def __init__(self, s, context=None):
        self.source, self.attr = s, {}
        self.match = m = self.match(s)
        if m and m.end(0):
            self.attr.update(self.process(m, context))
        if not self.attr:
            raise TokenError(s)

    def ast(self, tokens, stack, builder):
        tokens.append(self)

    def update_input_tokens(self, *tokens):
        pass

    @property
    def node_id(self):
        return self.get_expr

    @property
    def name(self):
        return self.attr.get('name', '')

    def set_expr(self, *tokens):
        self.attr['expr'] = self.name

    def __getattr__(self, item):
        if item.startswith('has_'):
            return item[4:] in self.attr
        elif item.startswith('get_'):
            return self.attr[item[4:]]

    def __repr__(self):
        return '{} <{}>'.format(self.name, self.__class__.__name__)

    def process(self, match, context=None):
        return {k: v for k, v in match.groupdict().items() if v is not None}

    def match(self, s):
        return self._re.match(s)
