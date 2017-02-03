#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
It provides formula parser class.
"""

import openpyxl
import os.path as osp
import schedula as sh
import schedula.utils as sh_utl
from .ranges import Ranges
from .cell import Cell, RangesAssembler
from .tokens.operand import _range2parts, Error
from .formulas.functions import flatten


def _get_name(name, names):
    if name not in names:
        name = name.upper()
        for n in names:
            if n.upper() == name:
                return n
    return name


class ExcelModel(object):
    def __init__(self):
        self.dsp = sh.Dispatcher()
        self.calculate = self.dsp.dispatch
        self.cells = {}

    @staticmethod
    def _yield_references(book, context=None):
        for n in book.defined_names.definedName:
            ref, i = n.name.upper(), n.localSheetId
            rng = Ranges().push(n.value, context=context).ranges[0]['name']
            sheet_names = book.sheetnames
            if i is not None:
                sheet_names = sheet_names[i:i + 1]
            for sn in sheet_names:
                name = _range2parts()(context, {'sheet': sn, 'ref': ref})
                yield name['name'], rng

    def loads(self, *file_names):
        return dict(map(self.load, file_names))

    def load(self, filename):
        book = openpyxl.load_workbook(filename, data_only=False)
        context = {'excel': osp.basename(filename)}
        references = dict(self._yield_references(book, context=context))
        self.pushes(*book.worksheets, context=context, references=references)
        return context['excel'], book

    def pushes(self, *worksheets, context=None, references=None):
        for ws in worksheets:
            self.push(ws, context=context, references=references)
        return self

    def push(self, worksheet, context=None, references=None):
        context = sh_utl.combine_dicts(
            context or {}, base={'sheet': worksheet.title}
        )
        f_refs = {
            k: v['ref'] for k, v in worksheet.formula_attributes.items()
            if v.get('t') == 'array' and 'ref' in v
        }
        f_rng = {Ranges().push(ref, context=context) for ref in f_refs.values()}
        for row in worksheet.iter_rows():
            for c in row:
                crd = c.coordinate
                crd = f_refs.get(crd, crd)

                cell = Cell(crd, c.value, context=context).compile()
                if cell.value is not sh_utl.EMPTY:
                    if any(not (cell.range - rng).ranges for rng in f_rng):
                        continue
                cell.update_inputs(references=references)

                if cell.add(self.dsp, context=context):
                    self.cells[cell.output] = cell
        return self

    def finish(self):
        for n_id in sorted(set(self.dsp.data_nodes) - set(self.cells)):
            if n_id is sh_utl.START:
                continue
            ra = RangesAssembler(n_id)
            for k, c in sorted(self.cells.items()):
                ra.push(c)
                if not ra.missing.ranges:
                    break

            self.dsp.add_function(None, ra, ra.inputs or None, [ra.output])

        return self

    def write(self, books=None, solution=None):
        books = {} if books is None else books
        solution = self.dsp.solution if solution is None else solution

        for r in solution.values():
            rng = r.ranges[0]
            filename, sheet_name = _get_name(rng['excel'], books), rng['sheet']

            if filename not in books:
                book = books[filename] = openpyxl.Workbook()
                for ws in book.worksheets:
                    book.remove_sheet(ws)
            else:
                book = books[filename]

            sheet_names = book.sheetnames
            sheet_name = _get_name(sheet_name, sheet_names)
            if sheet_name not in sheet_names:
                book.create_sheet(sheet_name)
            sheet = book[sheet_name]

            ref = '{c1}{r1}:{c2}{r2}'.format(**rng)
            for c, v in zip(flatten(sheet[ref], None), flatten(r.value, None)):
                if v is None:
                    v = Error.errors['#N/A']
                c.value = v

        return books
