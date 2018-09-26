#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
import os
import unittest
from setup import get_long_description, read_project_version

EXTRAS = os.environ.get('EXTRAS', 'all')


@unittest.skipIf(EXTRAS not in ('dev',), 'Not for extra %s.' % EXTRAS)
class TestSetup(unittest.TestCase):
    def test_long_description(self):
        self.assertTrue(bool(get_long_description()))

    def test_project_version(self):
        ver = read_project_version()
        from formulas import __version__
        self.assertEqual(ver, __version__)
