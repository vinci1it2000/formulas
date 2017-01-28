#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
It provides Operator classes.
"""

from . import Token
from .parenthesis import Parenthesis, _update_n_args
from ..errors import ParenthesesError
import regex
import collections


class Operator(Token):
    # http://office.microsoft.com/en-us/excel-help/calculation-operators-and-precedence-HP010078886.aspx
    _precedences = {
        ' ': 8, ',': 8, 'u-': 7, '%': 6, '^': 5, '*': 4, '/': 4, '+': 3, '-': 3,
        '&': 2, '=': 1, '<': 1, '>': 1, '<=': 1, '>=': 1, '<>': 1
    }
    _n_args = collections.defaultdict(lambda: 2)
    _n_args.update({'u-': 1, '%': 1})

    _re_process = None
    _strip = ' '

    def __repr__(self):
        return '{} <{}>'.format(self.name, Operator.__name__)

    def update_input_tokens(self, *tokens):
        if self.name in ' ,':
            self.attr['is_ranges'] = None
            for t in tokens:
                t.attr['is_ranges'] = None

    def set_expr(self, *tokens):
        expr, name = [t.get_expr for t in tokens], self.name
        if name == '%':
            expr = '{}%'.format(*expr)
        elif name == 'u-':
            expr = '-{}'.format(*expr)
        elif name in ' ,':
            expr = '(%s)' % ('%s ' % name.strip(' ')).join(expr)
        else:
            expr = '(%s)' % (' %s ' % name).join(expr)
        self.attr['expr'] = expr

    @property
    def get_n_args(self):
        return self._n_args[self.name]

    def process(self, match, context=None):
        if self._re_process:
            match = self._re_process.match(match.groups()[0].strip(self._strip))
        if match:
            return super(Operator, self).process(match, context=context)
        return {}

    @property
    def pred(self):
        return self._precedences[self.name]

    def update_name(self, tokens, stack):
        if self.name == '-':
            from .operand import Operand
            t = tokens[max(tokens.index(self) - 1, 0)]
            b = isinstance(t, Parenthesis) and t.has_end
            if not (b or isinstance(t, Operand)):
                self.attr['name'] = 'u-'
                _update_n_args(stack)

    def ast(self, tokens, stack, builder):
        super(Operator, self).ast(tokens, stack, builder)
        self.update_name(tokens, stack)
        pred = self.pred
        while stack and isinstance(stack[-1], Operator):
            if pred > stack[-1].pred:
                break
            builder.append(stack.pop())
        stack.append(self)

    def compile(self):
        from ..formulas.operators import OPERATORS
        return OPERATORS[self.name.upper()]


class Union(Operator):
    _re = regex.compile('^(?P<name>\s)\s*')


class Separator(Operator):
    _re = regex.compile('^([,\s]+)')
    _re_process = regex.compile('^(?P<name>,)$')

    def ast(self, tokens, stack, builder):
        super(Operator, self).ast(tokens, stack, builder)
        while stack and not stack[-1].has_start:
            builder.append(stack.pop())
        if not len(stack):
            raise ParenthesesError()


class OperatorToken(Operator):
    _re = regex.compile('^([\+\-\*\/\^&<>=\s]+|[\s%]+)')
    _re_process = regex.compile(
        '^(?P<name>(?P<sum_minus>[\+\-]+)|[\*\/\^&\%]|[<>]?=|[<>]|<>)$'
    )

    def process(self, match, context=None):
        attr = super(OperatorToken, self).process(match, context=context)
        if 'sum_minus' in attr:
            attr['name'] = '-+'[attr['sum_minus'].count('-') % 2 == 0]
        return attr
