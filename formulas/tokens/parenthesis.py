#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
It provides Parenthesis class.
"""

from . import Token
from ..errors import ParenthesesError, TokenError
# noinspection PyCompatibility
import regex


class Parenthesis(Token):
    _re = regex.compile(r'^\s*(?P<name>(?P<start>\()|(?P<end>\)))\s*')
    opens = {')': '('}
    n_args = 0

    def ast(self, tokens, stack, builder):
        from . operand import Operand
        if self.has_start and tokens and isinstance(tokens[-1], Operand):
            raise TokenError
        super(Parenthesis, self).ast(tokens, stack, builder)
        if self.has_start:
            stack.append(self)
            self.attr['check_n'] = self.attr.get('check_n', lambda t: t.n_args)
        else:
            while stack and not stack[-1].has_start:
                builder.append(stack.pop())

            if not stack or self.opens[self.name] != stack[-1].name:
                raise ParenthesesError()
            token = stack.pop()
            if not token.get_check_n(token):
                raise ParenthesesError()
            n = self.attr['n_args'] = token.n_args
            from .function import Function
            if stack and isinstance(stack[-1], Function):
                stack[-1].attr['n_args'] = token.n_args
                builder.append(stack.pop())
            elif n > 1:
                from .operator import Separator
                for i in range(n - 1):
                    builder.append(Separator(','))

            _update_n_args(stack)


def _update_n_args(stack):
    if stack:
        t = stack[-1]
        if isinstance(t, Parenthesis) and t.has_start:
            t.n_args += 1
