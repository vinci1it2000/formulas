#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2026 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
It provides formula parser class.
"""

# noinspection PyCompatibility
import regex
from .errors import TokenError, FormulaError, ParenthesesError
from .tokens.operand import String, Error, Number, Range, _col2index, _index2col
from .tokens.operator import OperatorToken, Separator, Intersect
from .tokens.function import Function, Array, Lambda
from .tokens.parenthesis import Parenthesis
from .builder import AstBuilder


_re_offset_literal = regex.compile(
    r"""
    OFFSET\(\s*
    (?P<ref>
        (?:'(?:''|[^'])+'!|[A-Za-z_][\w\.]*!)?
        \$?[A-Z]{1,3}\$?[1-9]\d*
        (?:\s*:\s*\$?[A-Z]{1,3}\$?[1-9]\d*)?
    )
    \s*,\s*(?P<rows>-?\d+)
    \s*,\s*(?P<cols>-?\d+)
    (?:\s*,\s*(?P<height>-?\d+))?
    (?:\s*,\s*(?P<width>-?\d+))?
    \s*\)
    """, regex.IGNORECASE | regex.X
)

_re_ref_split = regex.compile(
    r"""
    ^
    (?P<sheet>(?:'(?:''|[^'])+'|[A-Za-z_][\w\.]*)!)?
    \$?(?P<c1>[A-Z]{1,3})\$?(?P<r1>[1-9]\d*)
    (?:\s*:\s*\$?(?P<c2>[A-Z]{1,3})\$?(?P<r2>[1-9]\d*))?
    $
    """, regex.IGNORECASE | regex.X
)

#: Bounding-box size (rows, cols) used to load surrounding cells as
#: dependencies for OFFSET calls whose row/column offsets are not literal
#: integers. Larger values cover more dynamic targets at the cost of more
#: cells in the dependency graph.
DYNAMIC_OFFSET_BOUNDS = (50, 50)

_re_offset_dyn_base = regex.compile(
    r"""
    OFFSET\(\s*
    (?P<sheet>(?:'(?:''|[^'])+'!|[A-Za-z_][\w\.]*!)?)
    \$?(?P<c1>[A-Z]{1,3})\$?(?P<r1>[1-9]\d*)
    \s*,
    """, regex.IGNORECASE | regex.X
)


def _split_offset_args(rest):
    """Split the comma-separated arg tail of OFFSET; respects parens/quotes.

    Returns ``(args, consumed)`` where ``consumed`` is the number of chars
    of ``rest`` that made up ``OFFSET(...,...)``'s args plus the closing
    paren; or ``(None, 0)`` if the call is unclosed.
    """
    args, buf, depth, in_str = [], [], 0, False
    for i, ch in enumerate(rest):
        if in_str:
            buf.append(ch)
            if ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            buf.append(ch)
        elif ch == '(':
            depth += 1
            buf.append(ch)
        elif ch == ')':
            if depth == 0:
                args.append(''.join(buf).strip())
                return args, i + 1
            depth -= 1
            buf.append(ch)
        elif ch == ',' and depth == 0:
            args.append(''.join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    return None, 0


def _expand_dynamic_offset_bases(expr):
    """Auto-expand single-cell bases in OFFSET calls with non-literal offsets.

    After literal-arg OFFSETs have been resolved, any remaining
    ``OFFSET(<cell>, ...)`` is an indication that at least one offset is
    dynamic. Replace ``<cell>`` with a bounding-box range so the surrounding
    cells are loaded into the dependency graph, and ensure height/width are
    explicit (default to 1, 1) so the runtime function returns a scalar.
    """
    rows_bound, cols_bound = DYNAMIC_OFFSET_BOUNDS
    out, i = [], 0
    while i < len(expr):
        m = _re_offset_dyn_base.search(expr, pos=i)
        if not m:
            out.append(expr[i:])
            break
        out.append(expr[i:m.start()])
        sheet = m.group('sheet') or ''
        c1, r1 = m.group('c1').upper(), int(m.group('r1'))
        rest = expr[m.end():]
        args, consumed = _split_offset_args(rest)
        if args is None or len(args) < 2:
            out.append(expr[m.start():m.end()])
            i = m.end()
            continue
        n1 = _col2index(c1)
        n2, r2 = n1 + cols_bound - 1, r1 + rows_bound - 1
        box = f'{sheet}{c1}{r1}:{_index2col(n2)}{r2}'
        # Args in `args`: rows, cols, [height, width]
        # rows/cols stay as-is. If height/width missing, add 1, 1.
        if len(args) == 2:
            args = args + ['1', '1']
        new_call = f'OFFSET({box}, ' + ', '.join(args) + ')'
        out.append(new_call)
        i = m.end() + consumed
    return ''.join(out)


def _resolve_literal_offset(expr):
    """Rewrite OFFSET(<lit-ref>, <int>, <int>[, <int>[, <int>]]) at parse time.

    Returns the expression with all resolvable OFFSET calls replaced by
    direct range references. Unresolvable cases (non-literal args, out-of-
    bounds offsets) are left untouched or substituted with #REF!.
    """
    def _replace(match):
        m = _re_ref_split.match(match.group('ref').strip())
        if not m:
            return match.group(0)
        sheet = m.group('sheet') or ''
        c1 = m.group('c1').upper()
        r1 = int(m.group('r1'))
        c2 = (m.group('c2') or c1).upper()
        r2 = int(m.group('r2') or r1)
        rows = int(match.group('rows'))
        cols = int(match.group('cols'))
        h, w = match.group('height'), match.group('width')

        new_n1, new_r1 = _col2index(c1) + cols, r1 + rows
        if h is None:
            new_n2, new_r2 = _col2index(c2) + cols, r2 + rows
        else:
            h_i = int(h)
            w_i = int(w) if w is not None else (_col2index(c2) - _col2index(c1) + 1)
            if h_i <= 0 or w_i <= 0:
                return '#REF!'
            new_n2, new_r2 = new_n1 + w_i - 1, new_r1 + h_i - 1

        if new_n1 < 1 or new_r1 < 1 or new_n2 < 1 or new_r2 < 1:
            return '#REF!'

        new_c1, new_c2 = _index2col(new_n1), _index2col(new_n2)
        if new_c1 == new_c2 and new_r1 == new_r2:
            return f'{sheet}{new_c1}{new_r1}'
        return f'{sheet}{new_c1}{new_r1}:{new_c2}{new_r2}'

    prev = None
    while expr != prev:
        prev = expr
        expr = _re_offset_literal.sub(_replace, expr)
    return expr


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
            expr = _resolve_literal_offset(match['name'])
            expr = _expand_dynamic_offset_bases(expr)
            match['name'] = expr
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
