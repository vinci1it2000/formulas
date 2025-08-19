#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2025 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
It provides a custom Excel Reader class.
"""
from openpyxl.reader.excel import ExcelReader
from openpyxl.xml.constants import SHARED_STRINGS
from openpyxl.cell.text import Text
from openpyxl.xml.functions import iterparse
from openpyxl.xml.constants import SHEET_MAIN_NS
from ..functions.text import _re_hex


def replace_hex(match):
    return chr(int(match.group(1), 16))


def read_string_table(_raw_data, xml_source):
    """Read in all shared strings in the table"""

    strings = []
    STRING_TAG = '{%s}si' % SHEET_MAIN_NS

    for _, node in iterparse(xml_source):
        if node.tag == STRING_TAG:
            text = Text.from_tree(node).content
            if not _raw_data and '_x' in text:
                text = _re_hex.sub(replace_hex, text)
            text = text.replace('x005F_', '')
            node.clear()
            strings.append(text)

    return strings


class XlReader(ExcelReader):
    def __init__(self, *args, raw_date=True, _raw_data=False, **kwargs):
        super(XlReader, self).__init__(*args, **kwargs)
        self.raw_date, self._date_formats = raw_date, set()
        self._raw_data = _raw_data

    def read_worksheets(self):
        if self.raw_date:
            self._date_formats = self.wb._date_formats
            self.wb._date_formats = set()
        super(XlReader, self).read_worksheets()

    def read_strings(self):
        ct = self.package.find(SHARED_STRINGS)
        reader = read_string_table
        if ct is not None:
            strings_path = ct.PartName[1:]
            with self.archive.open(strings_path, ) as src:
                self.shared_strings = reader(self._raw_data, src)


def load_workbook(filename, _raw_data=False, **kw):
    if isinstance(filename, str) and filename.endswith('.ods'):
        from .ods_reader import ods_to_xlsx
        return ods_to_xlsx(filename, **kw)
    reader = XlReader(filename, _raw_data=_raw_data, **kw)
    reader.read()
    return reader.wb
