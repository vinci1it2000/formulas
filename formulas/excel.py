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
from .tokens.operand import _range2parts
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
            sheetnames = book.sheetnames if i is None else book.sheetnames[
                                                           i:i + 1]
            for sn in sheetnames:
                name = _range2parts()(context, {'sheet': sn, 'ref': ref})
                yield name['name'], rng

    def loads(self, *filenames):
        return dict(map(self.load, filenames))

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
        coord = {
            k: v['ref'] for k, v in worksheet.formula_attributes.items()
            if v.get('t') == 'array' and 'ref' in v
        }.get
        for row in worksheet.iter_rows():
            for c in row:
                crd = c.coordinate
                cell = Cell(coord(crd, crd), c.value, context=context).compile()
                cell.update_inputs(references=references)
                if cell.add(self.dsp, context=context):
                    self.cells[cell.output] = cell
        return self

    def finish(self):
        ras = []
        for k in set(self.dsp.data_nodes) - set(self.cells):
            try:
                ras.append(RangesAssembler(k))
            except ValueError:
                pass

        for ra in ras:
            for c in self.cells.values():
                ra.push(c)

            self.dsp.add_function(
                function=ra,
                inputs=ra.inputs or None,
                outputs=[ra.output]
            )
        return self

    def write(self, books=None, solution=None):
        books = {} if books is None else books
        solution = self.dsp.solution if solution is None else solution
        for k, v in solution.items():
            rng = v.ranges[0]
            filename, sheetname = _get_name(rng['excel'], books), rng['sheet']

            if filename not in books:
                books[filename] = openpyxl.Workbook()
            book = books[filename]
            sheetnames = book.sheetnames
            sheetname = _get_name(sheetname, sheetnames)
            if sheetname not in sheetnames:
                book.create_sheet(sheetname)
            sheet = book[sheetname]
            ref = '{c1}{r1}:{c2}{r2}'.format(**rng)
            for c, v in zip(flatten(sheet[ref], None), flatten(v.value, None)):
                c.value = v
        return books
