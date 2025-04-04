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
import regex
from openpyxl.reader.excel import ExcelReader
from openpyxl.xml.constants import SHARED_STRINGS
from openpyxl.reader.strings import read_rich_text
from openpyxl.cell.text import Text
from openpyxl.xml.functions import iterparse
from openpyxl.xml.constants import SHEET_MAIN_NS
from ..functions.text import HexValue, _re_hex


def read_string_table(xml_source):
    """Read in all shared strings in the table"""

    strings = []
    STRING_TAG = '{%s}si' % SHEET_MAIN_NS

    for _, node in iterparse(xml_source):
        if node.tag == STRING_TAG:
            text = Text.from_tree(node).content
            if text.startswith('_x') and _re_hex.match(text):
                text = HexValue(text)
            else:
                text = text.replace('x005F_', '')

            node.clear()
            strings.append(text)

    return strings


class XlReader(ExcelReader):
    def __init__(self, *args, raw_date=True, **kwargs):
        super(XlReader, self).__init__(*args, **kwargs)
        self.raw_date, self._date_formats = raw_date, set()

    def read_worksheets(self):
        if self.raw_date:
            self._date_formats = self.wb._date_formats
            self.wb._date_formats = set()
        super(XlReader, self).read_worksheets()

    def read_strings(self):
        ct = self.package.find(SHARED_STRINGS)
        reader = read_string_table
        if self.rich_text:
            reader = read_rich_text
        if ct is not None:
            strings_path = ct.PartName[1:]
            with self.archive.open(strings_path, ) as src:
                self.shared_strings = reader(src)


def load_workbook(filename, **kw):
    reader = XlReader(filename, **kw)
    reader.read()
    return reader.wb
