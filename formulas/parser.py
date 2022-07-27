#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
It provides formula parser class.
"""

# noinspection PyCompatibility
import regex
import typing as t

from .errors import TokenError, FormulaError, ParenthesesError
from .tokens import Token
from .tokens.operand import String, Error, Number, Range
from .tokens.operator import OperatorToken, Separator, Intersect
from .tokens.function import Function, Array
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
        Error, String, Number, Range, OperatorToken, Separator, Function, Array,
        Parenthesis, Intersect
    ]

    def is_formula(self, value: str) -> regex.Match:
        """Checks whether `value` is a formula.

        Args:
            value (str): The formula to test

        Returns:
            regex.Match: The match object.
        """        
        return self.formula_check.match(value) or Error._re.match(value)

    def ast(self, expression: str, context=None) -> tuple[t.Any, ...]:
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
                    token = f(expr, context)
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
            FormulaError(expression)
        builder.finish()
        return tokens, builder

    def compile(self, expression: str, context=None) -> t.Callable[..., t.Any]:
        """Shortcut to compile an `expression` to a callable.

        Example:

            .. codeblock:: python

                >>> from formulas import Parser
                >>> parser = Parser()
                >>> func = parser.compile('=A1+B1')

        Args:
            expression (str): The Excel formula to compile.
            context (_type_, optional): ...

        Returns:
            AstBuilder: The Formula Callable.
        """        
        _, builder = self.ast(expression, context)
        return builder.compile()

    def tokens(self, expression: str, context=None) -> t.List[t.Type[Token]]:
        """Shortcut to compile an `expression` to a callable.

        Example:

            .. codeblock:: python

                >>> from formulas import Parser
                >>> parser = Parser()
                >>> func = parser.compile('=A1+B1')

        Args:
            expression (str): The Excel formula to compile.
            context (_type_, optional): ...

        Returns:
            AstBuilder: The Formula Callable.
        """        
        tokens, _ = self.ast(expression, context)
        return tokens
