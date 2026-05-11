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


# Excel sheet limits — clamp targets so we never emit invalid refs.
_EXCEL_MAX_COL = 16384  # XFD
_EXCEL_MAX_ROW = 1048576

#: Bounding box (rows, cols) loaded as dependencies when an OFFSET call
#: has non-literal offsets. 16x16 = 256 cells per call covers the typical
#: dropdown-driven case while limiting graph bloat.
DYNAMIC_OFFSET_BOUNDS = (16, 16)

# Word-boundary-anchored to avoid clobbering user functions whose names
# end in OFFSET (e.g. SOFFSET).
_re_offset_literal = regex.compile(
    r"""
    \bOFFSET\(\s*
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

_re_offset_dyn_base = regex.compile(
    r"""
    \bOFFSET\(\s*
    (?P<sheet>(?:'(?:''|[^'])+'!|[A-Za-z_][\w\.]*!)?)
    \$?(?P<c1>[A-Z]{1,3})\$?(?P<r1>[1-9]\d*)
    \s*,
    """, regex.IGNORECASE | regex.X
)

# Matches a complete Excel string literal — `"..."` with `""` as the
# in-string escape for a literal quote.
_re_string_literal = regex.compile(r'"(?:""|[^"])*"')


def _strip_strings(expr):
    """Replace string literals with positional placeholders so subsequent
    regex passes never touch their contents."""
    lits = []

    def _stash(m):
        lits.append(m.group(0))
        return f'\x00{len(lits) - 1}\x00'

    return _re_string_literal.sub(_stash, expr), lits


def _restore_strings(expr, lits):
    if not lits:
        return expr
    return regex.sub(
        r'\x00(\d+)\x00', lambda m: lits[int(m.group(1))], expr
    )


def _split_offset_args(rest):
    """Split the comma-separated arg tail of OFFSET. Strings are already
    placeholder-substituted upstream, so we only track parens here.
    Returns ``(args, consumed)`` — ``consumed`` includes the closing paren —
    or ``(None, 0)`` for an unclosed call."""
    args, buf, depth = [], [], 0
    for i, ch in enumerate(rest):
        if ch == '(':
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
    """Expand single-cell base + add explicit (1, 1) size for OFFSET calls
    with non-literal offsets, so surrounding cells become dependencies."""
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
        args, consumed = _split_offset_args(expr[m.end():])
        if args is None or len(args) < 2:
            out.append(expr[m.start():m.end()])
            i = m.end()
            continue
        n1 = _col2index(c1)
        n2 = min(n1 + cols_bound - 1, _EXCEL_MAX_COL)
        r2 = min(r1 + rows_bound - 1, _EXCEL_MAX_ROW)
        box = f'{sheet}{c1}{r1}:{_index2col(n2)}{r2}'
        if len(args) == 2:
            args = args + ['1', '1']
        out.append(f'OFFSET({box}, ' + ', '.join(args) + ')')
        i = m.end() + consumed
    return ''.join(out)


def _resolve_literal_offset(expr):
    """Rewrite OFFSET(<lit-ref>, <int>, <int>[, <int>[, <int>]]) into a
    direct A1 reference; emit #REF! on out-of-bounds or invalid sizes."""
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

        if (new_n1 < 1 or new_r1 < 1
                or new_n2 > _EXCEL_MAX_COL or new_r2 > _EXCEL_MAX_ROW):
            return '#REF!'

        new_c1, new_c2 = _index2col(new_n1), _index2col(new_n2)
        if new_c1 == new_c2 and new_r1 == new_r2:
            return f'{sheet}{new_c1}{new_r1}'
        return f'{sheet}{new_c1}{new_r1}:{new_c2}{new_r2}'

    # Bounded fixpoint to handle nested OFFSETs without unbounded loops on
    # pathological input.
    for _ in range(8):
        new_expr = _re_offset_literal.sub(_replace, expr)
        if new_expr == expr:
            break
        expr = new_expr
    return expr


def _rewrite_offsets(expr):
    """Apply both OFFSET rewriters with string literals masked out so
    regex matches never escape into ``"..."`` content."""
    if 'OFFSET' not in expr and 'offset' not in expr:
        return expr
    stripped, lits = _strip_strings(expr)
    stripped = _resolve_literal_offset(stripped)
    stripped = _expand_dynamic_offset_bases(stripped)
    return _restore_strings(stripped, lits)


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
            expr = _rewrite_offsets(match['name'])
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
