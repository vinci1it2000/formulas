#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2021 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
It provides a custom Excel Reader class.
"""
from openpyxl.reader.excel import ExcelReader


class XlReader(ExcelReader):
    def __init__(self, *args, raw_date=True, **kwargs):
        super(XlReader, self).__init__(*args, **kwargs)
        self.raw_date, self._date_formats = raw_date, set()

    def read_worksheets(self):
        if self.raw_date:
            self._date_formats = self.wb._date_formats
            self.wb._date_formats = set()
        super(XlReader, self).read_worksheets()


def load_workbook(filename, **kw):
    reader = XlReader(filename, **kw)
    reader.read()
    return reader.wb
