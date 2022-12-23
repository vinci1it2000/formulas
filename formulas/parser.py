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

    def ast(self, expression: str, context=None) -> tuple[t.List[Token], AstBuilder]:
        """Main method to parse an `expression` to a callable.
        It uses the `AstBuilder` class to build the AST.
        Parses the expression into tokens, compiles the tokens to a callable using `Schedula`

        Example::

            .. codeblock :: python

                >>> from formulas import Parser
                >>> 
                >>> parser = Parser()
                >>> 
                >>> func_tokens = parser.ast('=A1+B1')[0]
                >>> func_callable = parser.ast('=A1+B1')[1].compile()
                >>> 
                >>> result = func_callable(A1=1, B1=2)
                >>> assert result == 3
                True


        Args:
            expression (str): The expression to parse.
            context (_type_, optional): _description_. Defaults to None.

        Raises:
            FormulaError: If the formula has a syntax error.
            ParenthesesError: _description_

        Returns:
            tuple[t.Tokens, AstBuilder]: A tuple with the tokens and the Builder.
        """        
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
            raise FormulaError(expression)
        builder.finish()
        return tokens, builder

    def compile(self, expression: str, context=None) -> t.Callable[..., t.Any]:
        """Shortcut to compile an `expression` to a callable.

        Example::

            .. codeblock:: python

                >>> from formulas import Parser
                >>> parser = Parser()
                >>> func = parser.compile('=A1+B1')
                >>> result = func(A1=2, B1=2)
                >>> asert result == 4
                True

        Args:
            expression (str): The Excel formula to compile.
            context (_type_, optional): ...

        Returns:
            AstBuilder: The Formula Callable.
        """        
        _, builder = self.ast(expression, context)
        return builder.compile()

    def tokens(self, expression: str, context=None) -> t.List[Token]:
        """Shortcut to identify the `expression` tokens.

        Example::

            .. codeblock:: python

                >>> from formulas import Parser
                >>> from formulas.token import Token
                >>>
                >>> parser = Parser()
                >>> tokens = parser.tokens('=A1+B1')
                >>> tokens
                [A1 <Range>, + <Operator>, B1 <Range>]
                >>> tokens[0].name
                'A1'
                >>> tokens[1].name
                '+'
                >>> tokens[2].name
                'B1'
                >>> isinstance(tokens[0], Token)
                True

        Args:
            expression (str): The Excel formula to compile.
            context (_type_, optional): ...

        Returns:
            List[Token]: A list of tokens.
        """        
        tokens, _ = self.ast(expression, context)
        return tokens
