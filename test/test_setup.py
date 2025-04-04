#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2025 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
import os
import sys
import unittest
import os.path as osp
sys.path.insert(0, osp.join(osp.dirname(__file__), '../'))
from setup import get_long_description, read_project_version

EXTRAS = os.environ.get('EXTRAS', 'dev')


def rst2html(source):
    from docutils.core import publish_string
    return publish_string(
        source, reader_name='standalone', parser_name='restructuredtext',
        writer_name='html', settings_overrides={'halt_level': 2}  # 2=WARN
    )[0]


@unittest.skipIf(EXTRAS not in ('dev',), 'Not for extra %s.' % EXTRAS)
class TestSetup(unittest.TestCase):
    def test_long_description(self):
        self.assertTrue(bool(rst2html(get_long_description())))

    def test_project_version(self):
        ver = read_project_version()
        from formulas import __version__
        self.assertEqual(ver, __version__)
