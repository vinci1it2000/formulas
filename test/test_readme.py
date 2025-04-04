#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2025 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
import os
import doctest
import unittest


EXTRAS = os.environ.get('EXTRAS', 'all')


@unittest.skipIf(EXTRAS not in ('all',), 'Not for extra %s.' % EXTRAS)
class TestReadme(unittest.TestCase):
    def test_readme(self):
        failure_count, test_count = doctest.testfile(
            '../README.rst', optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS
        )
        self.assertGreater(test_count, 0, (failure_count, test_count))
        self.assertEqual(failure_count, 0, (failure_count, test_count))
