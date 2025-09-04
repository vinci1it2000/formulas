#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2025 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
It provides formula parser class.
"""

# noinspection PyCompatibility
import regex
from .errors import TokenError, FormulaError, ParenthesesError
from .tokens.operand import String, Error, Number, Range
from .tokens.operator import OperatorToken, Separator, Intersect
from .tokens.function import Function, Array, Lambda
from .tokens.parenthesis import Parenthesis
from .builder import AstBuilder


class Parser:
    formula_check = regex.compile(
        r"""
        (?P<array>^\s*{\s*=\s*(?P<name>\S.*)\s*}\s*$)
        |
        (?P<value>^\s*=\s*(?P<name>\S.*))
        """, regex.IGNORECASE | regex.X | regex.DOTALL
    )
    ast_builder = AstBuilder
    filters = [
        Error, String, Number, Lambda, Range, OperatorToken, Separator,
        Function, Array, Parenthesis, Intersect
    ]

    def __init__(self, is_cell=False):
        self.is_cell = is_cell

    def is_formula(self, value):
        return self.formula_check.match(value) or Error._re.match(value)

    def ast(self, expression, context=None):
        try:
            match = self.is_formula(expression.replace('\n', '')).groupdict()
            expr = match['name']
        except (AttributeError, KeyError):
            raise FormulaError(expression)
        builder = self.ast_builder(match=match)
        filters, tokens, stack = self.filters, [], []
        Parenthesis('(').ast(tokens, stack, builder)
        while expr:
            for f in filters:
                try:
                    token = f(expr, context, self)
                    token.ast(tokens, stack, builder)
                    expr = expr[token.end_match:]
                    break
                except TokenError:
                    pass
                except FormulaError:
                    raise FormulaError(expression)
            else:
                raise FormulaError(expression)
        Parenthesis(')').ast(tokens, stack, builder)
        tokens = tokens[1:-1]
        while stack:
            if isinstance(stack[-1], Parenthesis):
                raise ParenthesesError()
            builder.append(stack.pop())
        if len(builder) != 1:
            raise FormulaError(expression)
        builder.finish()
        return tokens, builder
