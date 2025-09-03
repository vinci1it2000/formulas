#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2025 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
It provides Function classes.
"""

# noinspection PyCompatibility
import regex
import functools
import schedula as sh
from . import Token
from .parenthesis import Parenthesis
from .operand import Range
from .operator import Separator
from ..errors import TokenError, FormulaError


class Function(Token):
    _re = regex.compile(r'^\s*(?P<name>[A-Z_][\w\.]*)\(\s*', regex.IGNORECASE)

    def ast(self, tokens, stack, builder, check_n=lambda *args: True):
        super(Function, self).ast(tokens, stack, builder)
        stack.append(self)
        t = Parenthesis('(')
        t.attr['check_n'] = check_n
        t.ast(tokens, stack, builder)

    def compile(self):
        from formulas.functions import get_functions
        return get_functions()[self.name.upper()]

    def set_expr(self, *tokens):
        args = ', '.join(t.get_expr for t in tokens)
        self.attr['expr'] = '%s(%s)' % (self.name.upper(), args)


_re_lambda = regex.compile(r'''
^\s*(?P<name>(?:_XLFN\.)?LAMBDA|(?P<let>LET))\(\s*
(?:
    (?P<arg>                                # 1° argomento
        (?:
            [^()",]+                        # testo normale (niente " ( ) ,)
          | "(?:[^"]|"")*"                  # stringa Excel: "" è apice doppio escapato
          | (?P<paren1>                     # parentesi annidate
                \(
                    (?:
                        [^()"]+
                      | "(?:[^"]|"")*"
                      | (?P>paren1)
                    )*
                \)
            )
        )+
    )
    (?:\s*,\s*                              # , argomenti successivi
        (?P<arg>
            (?:
                [^()",]+
              | "(?:[^"]|"")*"
              | (?P<paren2>
                    \(
                        (?:
                            [^()"]+
                          | "(?:[^"]|"")*"
                          | (?P>paren2)
                        )*
                    \)
                )
            )+
        )
    )*
)?
\s*\)(?P<call>\()?
|
^\s*(?P<name>_XLETA\.)(?P<fun>[[:alpha:]_\\]+[[:alnum:]\.\_\\]*)
''', regex.IGNORECASE | regex.X)


class Lambda(Function):
    _re = _re_lambda

    def process(self, match, context=None, parser=None):
        attr = super(Lambda, self).process(match, context)

        if 'fun' not in attr:
            is_let = 'let' in attr
            arg = match.captures('arg')
            try:
                inp = [
                    Range(s, context, parser) for s in arg[:-1:1 + int(is_let)]
                ]
                if not len(arg) or any(
                        not t.attr.get('is_reference') for t in inp):
                    raise FormulaError()
            except TokenError:
                raise FormulaError()
            tkns, bld = parser.ast(f"={arg[-1]}", context=context)

            if is_let:
                for k, s in zip(inp, arg[1:-1:2]):
                    tk, bl = parser.ast(f"={s}", context=context)
                    tkns.extend(tk)
                    i = bl.get_node_id(bl[-1])
                    bld.dsp.add_function(
                        function_id=f"={i}",
                        function=sh.bypass,
                        inputs=[i],
                        outputs=[k.name],
                    )
                    bld.dsp.extend(bl.dsp)
            attr['func'] = {
                'inputs': [x.name for x in inp],
                'builder': bld,
                'tokens': tkns
            }
        return attr

    def ast(self, tokens, stack, builder, check_n=lambda *args: True):
        super(Lambda, self).ast(tokens, stack, builder, check_n)
        if 'fun' in self.attr:
            Parenthesis(')').ast(tokens, stack, builder)
        else:
            func = self.attr['func']
            inp_names = set(func['inputs'])
            it = {
                t.name: t for t in func['tokens']
                if isinstance(t, Range) and t.name not in inp_names
            }
            for i, token in enumerate(it.values()):
                if i:
                    Separator(',').ast(tokens, stack, builder)
                token.ast(tokens, stack, builder)
            if 'call' in self.attr:
                Separator(',').ast(tokens, stack, builder)
            else:
                Parenthesis(')').ast(tokens, stack, builder)

    def compile(self):
        from formulas.functions import get_functions
        if 'fun' in self.attr:
            func = get_functions()[self.attr['fun'].upper()]
            if isinstance(func, dict):
                func = func.copy()
                func['function'] = functools.partial(
                    get_functions()['LAMBDA'],
                    func=func['function'],
                    wrapper=True
                )
                return func
            return functools.partial(
                get_functions()['LAMBDA'],
                func=func,
                wrapper=True
            )
        else:
            attr_f = self.attr['func']
            func = attr_f['builder'].compile()
            if 'let' in self.attr:
                return func
            for x in attr_f['inputs']:
                func.inputs.pop(x, None)
                func.inputs[x] = None
            return functools.partial(
                get_functions()['LAMBDA'],
                func=func,
                wrapper='call' not in self.attr
            )

    def set_expr(self, *tokens):
        if 'fun' in self.attr:
            self.attr['expr'] = self.name.upper() + self.attr['fun'].upper()
        else:
            func = self.attr['func']
            bld = func['builder']
            calc = bld.get_node_id(bld[-1])
            inp = func['inputs']
            func = f"{self.name.upper()}({', '.join(inp + [calc])})"
            if 'call' in self.attr:
                func = f"{func}({', '.join(t.get_expr for t in tokens[-len(inp):])})"
            self.attr['expr'] = func


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
