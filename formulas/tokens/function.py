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
import collections
import schedula as sh
from . import Token
from .parenthesis import Parenthesis
from .operand import Range
from .operator import Separator
from ..errors import TokenError, FormulaError, FoundError
from formulas.functions import wrap_func


class Function(Token):
    _re = regex.compile(
        r'^\s*(?P<name>[A-Z_][\w\.]*)\(\s*(?P<call>\))?', regex.IGNORECASE
    )

    def _process(self, attr, match, context=None, parser=None):
        if getattr(parser, 'is_cell', False):
            from formulas.functions import get_functions
            fn = attr['name'].upper()
            if fn not in get_functions():
                attr['func_token'] = Range(fn, context, parser)
        return attr

    def process(self, match, context=None, parser=None):
        attr = super(Function, self).process(match, context, parser)
        return self._process(attr, match, context, parser)

    def _ast(self, tokens, stack, builder):
        has_call = self.has_call
        if self.has_func_token:
            fake = tokens[-2:]
            self.get_func_token.ast(fake, stack, builder)
            if not has_call:
                Separator(',').ast(fake, stack, builder)

        if has_call:
            Parenthesis(')').ast(tokens, stack, builder)

    def ast(self, tokens, stack, builder, check_n=lambda *args: True):
        super(Function, self).ast(tokens, stack, builder)
        stack.append(self)
        t = Parenthesis('(')
        t.attr['check_n'] = check_n
        t.ast(tokens, stack, builder)
        self._ast(tokens, stack, builder)

    def compile(self):
        from formulas.functions import get_functions
        if self.attr.get('func_token'):
            return run_function
        return get_functions()[self.name.upper()]

    def set_expr(self, *tokens):
        args = ', '.join(t.get_expr for t in tokens[int(self.has_func_token):])
        self.attr['expr'] = '%s(%s)' % (self.name.upper(), args)


_re_lambda = regex.compile(r'''
^\s*(?P<name>(?:_XLFN\.)?(?:LAMBDA|(?P<let>LET)))\(\s*
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
^\s*(?P<name>_XLETA\.)(?P<leta>[[:alpha:]_\\]+[[:alnum:]\.\_\\]*)
''', regex.IGNORECASE | regex.X)


class LambdaFunction(functools.partial):
    def __new__(cls, __func, *args, **kwargs):
        return super(LambdaFunction, cls).__new__(
            cls, wrap_func(__func), *args, **kwargs
        )

    def __repr__(self):
        from formulas.functions import Error
        return Error.errors['#VALUE!']

    def __call__(self, *args, **kwargs):
        if callable(self.func):
            return super().__call__(*args, **kwargs)
        from formulas.functions import Error
        raise FoundError(err=Error.errors['#NAME?'])


class LetaFunction(LambdaFunction):
    pass


@wrap_func
def run_function(fun, *args, **kwargs):
    if callable(fun):
        return fun(*args, **kwargs)
    from formulas.functions import Error
    raise FoundError(err=Error.errors['#NAME?'])


class Lambda(Function):
    _re = _re_lambda

    def _process(self, attr, match, context=None, parser=None):
        if 'leta' in attr:
            return attr
        tokens = []
        vars = []
        arg = match.captures('arg')
        separator = Separator(',')
        calc_tkns, calc = parser.ast(f"={arg[-1]}", context=context)
        expr = []
        ext_inp = {}
        try:
            if 'let' in attr:
                out_id = calc.get_node_id(calc[-1])
                valids = set(calc.dsp.shrink_dsp(outputs=[out_id]).data_nodes)
                for var, code in zip(arg[:-1:2], arg[1:-1:2]):
                    var = Range(var, context, parser)
                    vars.append(var)
                    tkns, bld = parser.ast(f"={code}", context=context)
                    tokens.extend((var, separator, *tkns, separator))
                    o_id = bld.get_node_id(bld[-1])
                    var_name = var.name
                    expr.extend((var_name, ', ', o_id, ', '))
                    if var_name in valids:
                        ext_inp.update(
                            (t.name, t) for t in tkns if isinstance(t, Range)
                        )
                        calc.dsp.add_function(
                            function_id=f"={o_id}",
                            function=sh.bypass,
                            inputs=[o_id],
                            outputs=[var_name],
                        )
                        calc.dsp.extend(bld.dsp)
            else:
                for var in arg[:-1]:
                    var = Range(var, context, parser)
                    vars.append(var)
                    tokens.extend((var, separator))
                    expr.extend((var.name, ', '))
        except TokenError:
            raise FormulaError()
        tokens.extend((*calc_tkns, Parenthesis(')')))
        expr.extend((calc.get_node_id(calc[-1]), ')'))
        ext_inp.update(
            (t.name, t) for t in calc_tkns if isinstance(t, Range)
        )
        for t in vars:
            if not t.get_is_reference:
                raise FormulaError()
            ext_inp.pop(t.name, None)

        if 'call' in attr:
            tokens.append(Parenthesis('('))
            expr.extend('(')
        attr['func'] = {
            'expr': ''.join(expr),
            'external_inputs': collections.OrderedDict(sorted(
                ext_inp.items()
            )),
            'tokens': tokens,
            'calculation': calc,
            'vars': [t.name for t in vars]
        }
        return attr

    def _ast(self, tokens, stack, builder):
        fake = tokens[-2:]
        if self.has_leta:
            tokens.pop()
            Parenthesis(')').ast(fake, stack, builder)
        else:
            func = self.get_func
            for i, token in enumerate(func['external_inputs'].values()):
                if i:
                    Separator(',').ast(fake, stack, builder)
                token.ast(fake, stack, builder)
            if self.has_call:
                Separator(',').ast(fake, stack, builder)
            else:
                Parenthesis(')').ast(fake, stack, builder)
            tokens.extend(func['tokens'])

    def compile(self):
        if self.has_leta:
            from formulas.functions import get_functions
            func = get_functions()[self.get_leta.upper()]
            if isinstance(func, dict):
                func = func.copy()
                func['function'] = functools.partial(
                    LetaFunction, func['function']
                )
                return func
            return functools.partial(LetaFunction, func)
        else:
            attr = self.get_func
            func = attr['calculation'].compile()
            if self.has_let:
                return func
            for x in attr['vars']:
                func.inputs.pop(x, None)
                func.inputs[x] = None

            if self.has_call:
                return wrap_func(func)
            return functools.partial(LambdaFunction, func)

            for x in attr['vars']:
                func.inputs.pop(x, None)
                func.inputs[x] = None
            return functools.partial(_lambda, func, wrapper=not self.has_call)

    def set_expr(self, *tokens):
        func = self.name.upper()
        if self.has_leta:
            func = f"{func}{self.get_leta.upper()}"
        else:
            d = self.get_func
            func = f"{func}({d['expr']}"
            if self.has_call:
                n = len(d['external_inputs'])
                func = f"{func}{', '.join(t.get_expr for t in tokens[n:])})"
        self.attr['expr'] = func


def _check_tkn_n_args(n_args, token):
    return token.n_args == n_args


class Array(Function):
    _re = regex.compile(r'^\s*(?P<name>(?P<start>{)|(?P<end>})|(?P<sep>;))\s*')

    def _process(self, attr, match, context=None, parser=None):
        return attr

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
