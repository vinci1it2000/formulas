#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
It provides Function classes.
"""

# noinspection PyCompatibility
import regex
import functools
from . import Token
from .parenthesis import Parenthesis


class Function(Token):
    _re = regex.compile(r'^\s*@?(?P<name>[A-Z_][\w\.]*)\(\s*', regex.IGNORECASE)

    def ast(self, tokens, stack, builder, check_n=lambda *args: True):
        super(Function, self).ast(tokens, stack, builder)
        stack.append(self)
        t = Parenthesis('(')
        t.attr['check_n'] = check_n
        t.ast(tokens, stack, builder)

    def compile(self):
        from ..formulas.functions import get_functions
        return get_functions()[self.name.upper()]

    def set_expr(self, *tokens):
        args = ', '.join(t.get_expr for t in tokens)
        self.attr['expr'] = '%s(%s)' % (self.name.upper(), args)


def _check_tkn_n_args(n_args, token):
    return token.n_args == n_args


class Array(Function):
    _re = regex.compile(r'^\s*(?P<name>(?P<start>{)|(?P<end>})|(?P<sep>;))\s*')

    def ast(self, tokens, stack, builder, check_n=lambda t: t.n_args):
        if self.has_start:
            Function('ARRAY(').ast(tokens, stack, builder, check_n=check_n)
            Function('ARRAY(').ast(tokens, stack, builder, check_n=check_n)
        else:
            token = Parenthesis(')')
            token.ast(tokens, stack, builder)
            if self.has_sep:
                check_n = functools.partial(_check_tkn_n_args, token.get_n_args)
                Function('ARRAY(').ast(tokens, stack, builder, check_n=check_n)
            else:
                Parenthesis(')').ast(tokens, stack, builder)
