#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2026 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
import os
import unittest
import warnings

import formulas

PACKAGE = os.path.dirname(formulas.__file__)


class TestSource(unittest.TestCase):
    def test_modules_compile_without_syntax_warnings(self):
        # Python 3.12+ warns on `is` with a literal (`x is 1.0`): identity
        # with a constant depends on how the interpreter shares objects.
        for root, _, files in os.walk(PACKAGE):
            for name in sorted(f for f in files if f.endswith('.py')):
                path = os.path.join(root, name)
                with open(path, encoding='utf-8') as f, self.subTest(path):
                    source = f.read()
                    with warnings.catch_warnings():
                        warnings.simplefilter('error', SyntaxWarning)
                        compile(source, path, 'exec')
