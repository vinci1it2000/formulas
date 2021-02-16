#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2021 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
It provides Excel model class.

Sub-Modules:

.. currentmodule:: formulas.excel

.. autosummary::
    :nosignatures:
    :toctree: excel/

    ~cycle
    ~xlreader
"""
import os
import logging
import functools
import numpy as np
import os.path as osp
import schedula as sh
from ..ranges import Ranges
from ..functions import flatten
from ..errors import InvalidRangeName
from ..cell import Cell, RangesAssembler, Ref, CellWrapper
from ..tokens.operand import XlError, _re_sheet_id, _re_build_id

log = logging.getLogger(__name__)
BOOK = sh.Token('Book')
SHEETS = sh.Token('Sheets')
CIRCULAR = sh.Token('CIRCULAR')


class XlCircular(XlError):
    def __str__(self):
        return '0'


ERR_CIRCULAR = XlCircular('#CIRC!')


def _get_name(name, names):
    if name not in names:
        name = name.upper()
        for n in names:
            if n.upper() == name:
                return n
    return name


class ExcelModel:
    compile_class = sh.DispatchPipe

    def __init__(self):
        self.dsp = sh.Dispatcher(name='ExcelModel')
        self.cells = {}
        self.books = {}
        self.basedir = None

    def calculate(self, *args, **kwargs):
        return self.dsp.dispatch(*args, **kwargs)

    def __getstate__(self):
        return {'dsp': self.dsp, 'cells': {}, 'books': {}}

    def _update_refs(self, nodes, refs):
        dsp = self.dsp.get_sub_dsp(nodes)
        dsp.raises = ''
        sol = dsp(dict.fromkeys(set(dsp.data_nodes) - set(refs), sh.EMPTY))
        refs.update({
            k: v for k, v in sol.items() if k in refs and isinstance(v, Ranges)
        })

    def add_references(self, book, context=None):
        refs, nodes = {}, set()
        for n in book.defined_names.definedName:
            ref = Ref(n.name.upper(), '=%s' % n.value, context).compile(
                context=context
            )
            nodes.update(ref.add(self.dsp, context=context))
            refs[ref.output] = None
        self._update_refs(nodes, refs)
        return refs

    def loads(self, *file_names):
        for filename in file_names:
            self.load(filename)
        return self

    def load(self, filename):
        book, context = self.add_book(filename)
        self.pushes(*book.worksheets, context=context)
        return self

    def from_ranges(self, *ranges):
        return self.complete(ranges)

    def pushes(self, *worksheets, context=None):
        for ws in worksheets:
            self.push(ws, context=context)
        return self

    def push(self, worksheet, context):
        worksheet, context = self.add_sheet(worksheet, context)
        references = self.references
        formula_references = self.formula_references(context)
        formula_ranges = self.formula_ranges(context)
        external_links = self.external_links(context)

        for row in worksheet.iter_rows():
            for c in row:
                if hasattr(c, 'value'):
                    self.add_cell(
                        c, context, references=references,
                        formula_references=formula_references,
                        formula_ranges=formula_ranges,
                        external_links=external_links
                    )
        return self

    def add_book(self, book=None, context=None, data_only=False):
        ctx = (context or {}).copy()
        are_in, get_in = sh.are_in_nested_dicts, sh.get_nested_dicts
        if isinstance(book, str):
            ctx['directory'], ctx['filename'] = osp.split(book)
        if self.basedir is None:
            self.basedir = osp.abspath(ctx['directory'] or '.')
        if ctx['directory']:
            ctx['directory'] = osp.relpath(ctx['directory'], self.basedir)
        if ctx['directory'] == '.':
            ctx['directory'] = ''
        fpath = osp.join(ctx['directory'], ctx['filename'])
        ctx['excel'] = fpath.upper()
        data = get_in(self.books, ctx['excel'])
        book = data.get(BOOK)
        if not book:
            from .xlreader import load_workbook
            data[BOOK] = book = load_workbook(
                osp.join(self.basedir, fpath), data_only=data_only
            )

        if 'external_links' not in data:
            fdir = osp.join(self.basedir, ctx['directory'])
            data['external_links'] = {
                str(i + 1): osp.split(osp.relpath(osp.realpath(osp.join(
                    fdir, el.file_link.Target
                )), self.basedir))
                for i, el in enumerate(book._external_links)
                if el.file_link.Target.endswith('.xlsx')
            }

        if 'references' not in data:
            data['references'] = self.add_references(
                book, context=sh.combine_dicts(ctx, base=sh.selector(
                    ('external_links',), data, allow_miss=True
                ))
            )

        return book, ctx

    def add_sheet(self, worksheet, context):
        get_in = sh.get_nested_dicts
        if isinstance(worksheet, str):
            book = get_in(self.books, context['excel'], BOOK)
            worksheet = book[_get_name(worksheet, book.sheetnames)]

        ctx = {'sheet': worksheet.title.upper()}
        ctx.update(context)

        d = get_in(self.books, ctx['excel'], SHEETS, ctx['sheet'])
        if 'formula_references' not in d:
            d['formula_references'] = formula_references = {
                k: v['ref'] for k, v in worksheet.formula_attributes.items()
                if v.get('t') == 'array' and 'ref' in v
            }
        else:
            formula_references = d['formula_references']

        if 'formula_ranges' not in d:
            d['formula_ranges'] = {
                Ranges().push(ref, context=ctx)
                for ref in formula_references.values()
            }
        return worksheet, ctx

    @property
    def references(self):
        return sh.combine_dicts(*(
            d.get('references', {}) for d in self.books.values()
        ))

    def formula_references(self, ctx):
        return sh.get_nested_dicts(
            self.books, ctx['excel'], SHEETS, ctx['sheet'], 'formula_references'
        )

    def formula_ranges(self, ctx):
        return sh.get_nested_dicts(
            self.books, ctx['excel'], SHEETS, ctx['sheet'], 'formula_ranges'
        )

    def external_links(self, ctx):
        return sh.get_nested_dicts(self.books, ctx['excel'], 'external_links')

    def add_cell(self, cell, context, references=None, formula_references=None,
                 formula_ranges=None, external_links=None):
        if formula_references is None:
            formula_references = self.formula_references(context)

        if formula_ranges is None:
            formula_ranges = self.formula_ranges(context)

        if references is None:
            references = self.references

        if external_links is None:
            external_links = self.external_links(context)

        ctx = {'external_links': external_links}
        ctx.update(context)
        crd = cell.coordinate
        crd = formula_references.get(crd, crd)
        val = cell.value
        val = cell.data_type == 'f' and val[:2] == '==' and val[1:] or val
        check_formula = cell.data_type != 's'
        cell = Cell(crd, val, context=ctx, check_formula=check_formula).compile(
            references=references, context=ctx
        )
        if cell.output in self.cells:
            return
        if cell.value is not sh.EMPTY:
            if any(not (cell.range - rng).ranges for rng in formula_ranges):
                return

        if cell.add(self.dsp, context=ctx):
            self.cells[cell.output] = cell
            return cell

    def complete(self, stack=None):
        if stack is None:
            stack = set(self.dsp.data_nodes) - set(self.cells)
            stack -= set(self.references)
        stack = sorted(stack)
        while stack:
            n_id = stack.pop()
            if isinstance(n_id, sh.Token):
                continue
            try:
                rng = Ranges().push(n_id).ranges[0]
            except InvalidRangeName:  # Missing Reference.
                log.warning('Missing Reference `{}`!'.format(n_id))
                Ref(n_id, '=#REF!').compile().add(self.dsp)
                continue
            book = osp.join(rng.get('directory', ''),
                            rng.get('filename', rng.get('excel_id', '')))

            try:
                context = self.add_book(book)[1]
                worksheet, context = self.add_sheet(rng['sheet'], context)
            except Exception as ex:  # Missing excel file or sheet.
                log.warning('Error in loading `{}`:\n{}'.format(n_id, ex))
                Cell(n_id, '=#REF!').compile().add(self.dsp)
                self.books.pop(book)
                continue

            references = self.references
            formula_references = self.formula_references(context)
            formula_ranges = self.formula_ranges(context)
            external_links = self.external_links(context)
            rng = '{c1}{r1}:{c2}{r2}'.format(**rng)
            for c in flatten(worksheet[rng], None):
                if hasattr(c, 'value'):
                    cell = self.add_cell(
                        c, context, references=references,
                        formula_references=formula_references,
                        formula_ranges=formula_ranges,
                        external_links=external_links
                    )
                    if cell:
                        stack.extend(cell.inputs or ())
        return self

    def _assemble_ranges(self, cells, nodes=None):
        get = sh.get_nested_dicts
        pred = self.dsp.dmap.pred
        if nodes is None:
            nodes = set(self.dsp.data_nodes).difference(self.dsp.default_values)
        it = (
            k for k in nodes
            if not pred[k] and not isinstance(k, sh.Token)
        )
        for n_id in it:
            if isinstance(n_id, sh.Token):
                continue
            try:
                ra = RangesAssembler(n_id)
            except ValueError:
                continue
            rng = ra.range.ranges[0]
            for output, indices in get(cells, rng['sheet_id'], default=list):
                if not ra.push(output, indices):
                    break
            ra.add(self.dsp)

    def assemble(self):
        cells, get = {}, sh.get_nested_dicts
        for c in self.cells.values():
            rng = c.range.ranges[0]
            get(cells, rng['sheet_id'], default=list).append((
                c.output, RangesAssembler._range_indices(c.range)
            ))
        self._assemble_ranges(cells)
        return self

    def finish(self, complete=True, circular=False, assemble=True):
        if complete:
            self.complete()
        if assemble:
            self.assemble()
        if circular:
            self.solve_circular()
        return self

    def to_dict(self):
        nodes = {
            k: d['value']
            for k, d in self.dsp.default_values.items()
            if not isinstance(k, sh.Token)
        }
        nodes = {
            k: isinstance(v, str) and v.startswith('=') and '="%s"' % v or v
            for k, v in nodes.items() if v != [[sh.EMPTY]]
        }
        for d in self.dsp.function_nodes.values():
            fun = d['function']
            if isinstance(fun, CellWrapper):
                nodes.update(dict.fromkeys(d['outputs'], fun.__name__))
        return nodes

    def from_dict(self, adict, context=None, assemble=True):
        refs, cells, nodes, get = {}, {}, set(), sh.get_nested_dicts
        for k, v in adict.items():
            try:
                cell = Cell(k, v, context=context)
            except ValueError:
                ref = Ref(k, v, context=context).compile(context=context)
                nodes.update(ref.add(self.dsp))
                refs[ref.output] = None
                continue
            cells[cell.output] = cell
        self._update_refs(nodes, refs)
        for cell in cells.values():
            nodes.update(cell.compile(references=refs).add(self.dsp))
        self.cells.update(cells)
        if assemble:
            self.assemble()
        return self

    def write(self, books=None, solution=None, dirpath=None):
        books = {} if books is None else books
        solution = self.dsp.solution if solution is None else solution
        are_in, get_in = sh.are_in_nested_dicts, sh.get_nested_dicts
        for k, r in solution.items():
            if isinstance(k, sh.Token):
                continue
            if isinstance(r, Ranges):
                rng = {k: v for k, v in _re_sheet_id.match(
                    r.ranges[0]['sheet_id']
                ).groupdict().items() if v is not None}
                rng.update(r.ranges[0])
            else:
                try:
                    r = Ranges().push(k, r)
                    rng = r.ranges[0]
                except ValueError:  # Reference.
                    rng = {'sheet': ''}
            fpath = osp.join(rng.get('directory', ''), rng.get('filename', ''))
            fpath, sheet_name = _get_name(fpath, books), rng.get('sheet')
            if not (fpath and sheet_name):
                log.info('Node `%s` cannot be saved '
                         '(missing filename and/or sheet_name).' % k)
                continue
            elif _re_build_id.match(fpath):
                continue
            if not are_in(books, fpath, BOOK):
                from openpyxl import Workbook
                book = get_in(books, fpath, BOOK, default=Workbook)
                for ws in book.worksheets:
                    book.remove(ws)
            else:
                book = books[fpath][BOOK]

            sheet_names = book.sheetnames
            sheet_name = _get_name(sheet_name, sheet_names)
            if sheet_name not in sheet_names:
                book.create_sheet(sheet_name)
            sheet = book[sheet_name]

            ref = '{c1}{r1}:{c2}{r2}'.format(**rng)
            for c, v in zip(flatten(sheet[ref], None), flatten(r.value, None)):
                try:
                    if v is sh.EMPTY:
                        v = None
                    elif isinstance(v, np.generic):
                        v = v.item()
                    elif isinstance(v, XlError):
                        v = str(v)
                    c.value = v
                    if c.data_type == 'f':
                        c.data_type = 's'
                except AttributeError:
                    pass
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
            for fpath, d in books.items():
                d[BOOK].save(osp.join(dirpath, fpath))
        return books

    def compile(self, inputs, outputs):
        dsp = self.dsp.shrink_dsp(outputs=outputs)

        dsp.default_values = sh.selector(
            set(dsp.default_values) - set(inputs), dsp.default_values
        )

        res = dsp()

        dsp = dsp.get_sub_dsp_from_workflow(
            outputs, graph=dsp.dmap, reverse=True, blockers=res,
            wildcard=False
        )

        keys = set(dsp.data_nodes) - set(dsp.default_values)
        for k, v in sh.selector(keys, res, allow_miss=True).items():
            dsp.set_default_value(k, v.value)

        func = self.compile_class(
            dsp=dsp,
            function_id=self.dsp.name,
            inputs=inputs,
            outputs=outputs
        )

        return func

    def solve_circular(self):
        from .cycle import simple_cycles
        from collections import Counter
        mod, dsp = {}, self.dsp
        f_nodes, d_nodes, dmap = dsp.function_nodes, dsp.data_nodes, dsp.dmap
        cycles = list(simple_cycles(dmap.succ))
        cycles_nodes = Counter(sum(cycles, []))
        for cycle in sorted(map(set, cycles)):
            cycles_nodes.subtract(cycle)
            active_nodes = {k for k, v in cycles_nodes.items() if v}
            for k in sorted(cycle.intersection(f_nodes)):
                if _check_cycles(dmap, k, f_nodes, cycle, active_nodes, mod):
                    break
            else:
                cycles_nodes.update(cycle)
                dist = sh.inf(len(cycle) + 1, 0)
                for k in sorted(cycle.intersection(d_nodes)):
                    dsp.set_default_value(k, ERR_CIRCULAR, dist)

        if mod:  # Update dsp.
            dsp.add_data(CIRCULAR, ERR_CIRCULAR)

            for k, v in mod.items():
                d = f_nodes[k]
                d['inputs'] = [CIRCULAR if i in v else i for i in d['inputs']]
                dmap.remove_edges_from(((i, k) for i in v))
                dmap.add_edge(CIRCULAR, k)

        return self


def _check_range_all_cycles(nodes, active_nodes, j):
    if isinstance(nodes[j]['function'], RangesAssembler):
        return active_nodes.intersection(nodes[j]['inputs'])
    return False


def _check_cycles(dmap, node_id, nodes, cycle, active_nodes, mod=None):
    node, mod = nodes[node_id], {} if mod is None else mod
    _map = dict(zip(node['function'].inputs, node['inputs']))
    pred, res = dmap.pred, ()
    check = functools.partial(_check_range_all_cycles, nodes, active_nodes)
    if not any(any(map(check, pred[k])) for k in _map.values() if k in cycle):
        cycle = [i for i, j in _map.items() if j in cycle]
        try:
            res = tuple(map(_map.get, node['function'].check_cycles(cycle)))
            res and sh.get_nested_dicts(mod, node_id, default=set).update(res)
        except AttributeError:
            pass
    return res
