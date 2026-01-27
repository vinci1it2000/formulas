#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2025 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
It provides Operator classes.
"""

import collections

# noinspection PyCompatibility
import regex

from ..errors import FormulaError, ParenthesesError
from . import Token
from .parenthesis import Parenthesis, _update_n_args


class Operator(Token):
    # http://office.microsoft.com/en-us/excel-help/calculation-operators-and-
    # precedence-HP010078886.aspx
    _precedences = {
        ':': 8, ' ': 8, ',': 8, 'u-': 7, 'u+': 7, '@': 7, '%': 6, '^': 5,
        '*': 4, '/': 4, '+': 3, '-': 3, '&': 2, '=': 1, '<': 1, '>': 1, '<=': 1,
        '>=': 1, '<>': 1
    }
    _n_args = collections.defaultdict(lambda: 2)
    _n_args.update({'u-': 1, 'u+': 1, '@': 1, '%': 1})

    _re_process = None
    _replace = ' '

    def __repr__(self):
        return f'{self.name} <{Operator.__name__}>'

    def update_input_tokens(self, *tokens):
        if self.name in ' ,:':
            self.attr['is_ranges'] = True
            from .function import Function
            from .operand import Error, Range
            for t in tokens:
                if isinstance(t, Range) or isinstance(t, Function) and self.name in ':' or isinstance(t, Error) and t.name == '#REF!':
                    t.attr['is_ranges'] = True
                elif not t.attr.get('is_ranges', False):
                    raise FormulaError()
        else:
            super().update_input_tokens(*tokens)

    def set_expr(self, *tokens):
        expr, name = [t.get_expr for t in tokens], self.name
        if name == '%':
            expr = '{}%'.format(*expr)
        elif name in ('u-', 'u+'):
            expr = '{}{}'.format(name[1], *expr)
        elif name == '@':
            expr = '@{}'.format(*expr)
        elif name in ' ,:':
            expr = '({})'.format(('{} '.format(name.strip(' '))).join(expr))
        else:
            expr = '({})'.format((f' {name} ').join(expr))
        self.attr['expr'] = expr

    @property
    def get_n_args(self):
        return self._n_args[self.name]

    def process(self, match, context=None, parser=None):
        if self._re_process:
            s = match.groups()[0].replace(self._replace, '')
            match = self._re_process.match(s)
        if match:
            return super().process(
                match, context=context, parser=parser
            )
        return {}

    @property
    def pred(self):
        return self._precedences[self.name]

    def update_name(self, tokens, stack):
        if self.name in '-+':
            from .operand import Operand
            t = tokens[max(tokens.index(self) - 1, 0)]
            b = isinstance(t, Parenthesis) and t.has_end
            b |= isinstance(t, Operator) and t.name == '%'
            if not (b or isinstance(t, Operand)):
                self.attr['name'] = f'u{self.name}'
                _update_n_args(stack)
        elif self.name == '@':
            from .operand import Operand
            t = tokens[max(tokens.index(self) - 1, 0)]
            b = isinstance(t, Parenthesis) and t.has_end
            b |= isinstance(t, Operator) and t.name == '%'
            if not (b or isinstance(t, Operand)):
                self.attr['name'] = f'{self.name}'
                _update_n_args(stack)

    def ast(self, tokens, stack, builder):
        super().ast(tokens, stack, builder)
        self.update_name(tokens, stack)
        pred = self.pred
        while stack and isinstance(stack[-1], Operator):
            if pred > stack[-1].pred:
                break
            builder.append(stack.pop())
        stack.append(self)

    def compile(self):
        from ..functions.operators import OPERATORS
        return OPERATORS[self.name.upper()]


class Intersect(Operator):
    _re = regex.compile(r'^(?P<name>\s)\s*')


class Separator(Operator):
    _re = regex.compile(r'^(\s*,\s*)')
    _re_process = regex.compile(r'^\s*(?P<name>,)$')

    def ast(self, tokens, stack, builder):
        if tokens:
            lt = tokens[-1]
            from .operand import String
            if isinstance(lt, Separator) or (
                    lt.get_name == '(' and not isinstance(lt, String)
            ):
                from .operand import Empty
                Empty().ast(tokens, stack, builder)
        super(Operator, self).ast(tokens, stack, builder)
        while stack and not stack[-1].has_start:
            builder.append(stack.pop())
        if not len(stack):
            raise ParenthesesError()


class OperatorToken(Operator):
    _re = regex.compile(
        r'^(\s*([<>]=|<>|[\*\/\^&<>=])(?=\s*[\+\-])|\s*%+|[\+\-\*\/\^&<>=\s:@]+)'
    )
    _re_process = regex.compile(
        r'^\s*(?P<name>(?P<sum_minus>[\+\s\-]+)|[<>]?=|<>|[\*\/\^&\%:<>@])$'
    )

    def process(self, match, context=None, parser=None):
        attr = super().process(match, context=context)
        if 'sum_minus' in attr:
            attr['name'] = '-+'[attr['sum_minus'].count('-') % 2 == 0]
        return attr
