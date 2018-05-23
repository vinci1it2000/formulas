#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

import unittest
import ddt
import numpy as np
from formulas.tokens.operand import Error
from formulas.ranges import Ranges
from formulas.errors import RangeValueError


@ddt.ddt
class TestOperators(unittest.TestCase):
    @ddt.data(
        ((('D7:F14',), ('C:G',)), '<Ranges>()'),
        ((('D7:F14',), ('C:E',)), '<Ranges>(F7:F14)'),
        ((('D7:F14',), ('C4:E9',)), '<Ranges>(F7:F14, D10:E14)'),
        ((('I7:L14',), ('H9:M12',)), '<Ranges>(I7:L8, I13:L14)'),
        ((('I7:L14',), ('J5:K16',)), '<Ranges>(I7:I14, L7:L14)'),
        ((('F24:I32',), ('G20:H26',)), '<Ranges>(F24:F32, I24:I32, G27:H32)'),
        ((('M23:P30',), ('L20:Q25',)), '<Ranges>(M26:P30)'),
        ((('M23:P30', 'L20:L25', 'Q20:Q25', 'M20:P22',), ('L20:Q25',)),
         '<Ranges>(M26:P30)'),
        ((('M23:P30', 'L20:L25', 'Q20:Q25', 'M20:P22',),
          ('P18:R27', 'L25:N32')),
         '<Ranges>(O23:O30, M23:N24, P28:P30, L20:L24, M20:O22)'))
    def test_sub_ranges(self, case):
        (range1, range2), result = case
        output = str(Ranges().pushes(range1) - Ranges().pushes(range2))
        self.assertEqual(result, output, '%s != %s' % (result, output))

    @ddt.data(
        ((('D7:F14',), ('C:E',)), '<Ranges>(D7:F14, C:C, D:E6, D15:E)'),
        ((('D7:F14',), ('C4:E9',)), '<Ranges>(D7:F14, C4:C9, D4:E6)'),
        ((('I7:L14',), ('H9:M12',)), '<Ranges>(I7:L14, H9:H12, M9:M12)'),
        ((('I7:L14',), ('J5:K16',)), '<Ranges>(I7:L14, J5:K6, J15:K16)'),
        ((('F24:I32',), ('G20:H26',)), '<Ranges>(F24:I32, G20:H23)'),
        ((('M23:P30',), ('L20:Q25',)),
         '<Ranges>(M23:P30, L20:L25, Q20:Q25, M20:P22)'),
        ((('M23:P30', 'L20:L25', 'Q20:Q25', 'M20:P22',), ('L20:Q25',)),
         '<Ranges>(M23:P30, L20:L25, Q20:Q25, M20:P22)'),
        ((('M23:P30', 'L20:L25', 'Q20:Q25', 'M20:P22',),
          ('P18:R27', 'L25:N32')),
         '<Ranges>(M23:P30, L20:L25, Q20:Q25, M20:P22, R18:R27, Q18:Q19,'
         ' Q26:Q27, P18:P19, L26:L32, M31:N32)'))
    def test_union_ranges(self, case):
        (range1, range2), result = case
        output = str(Ranges().pushes(range1) | Ranges().pushes(range2))
        self.assertEqual(result, output, '%s != %s' % (result, output))

    @ddt.data(
        ((('D7:F14',), ('C:E',)), '<Ranges>(C:F)'),
        ((('D7:F14',), ('C4:E9',)), '<Ranges>(C4:F14)'),
        ((("'[xl.xls]s1'!I7:L14",), ("'[xl.xls]s1'!H9:M12",)),
         "<Ranges>('[XL.XLS]S1'!H7:M14)"),
        ((('I7:L14',), ('J5:K16',)), '<Ranges>(I5:L16)'),
        ((('F24:I32',), ('G20:H26',)), '<Ranges>(F20:I32)'),
        ((('M23:P30',), ('L20:Q25',)), '<Ranges>(L20:Q30)'),
        ((('M23:P30', 'L20:L25', 'Q20:Q25', 'M20:P22',), ('L20:Q25',)),
         '<Ranges>(L20:Q30)'),
        ((('M23:P30', 'L20:L25', 'Q20:Q25', 'M20:P22',),
          ('P18:R27', 'L25:N32')), '<Ranges>(L18:R32)'))
    def test_add_ranges(self, case):
        (range1, range2), result = case
        output = str(Ranges().pushes(range1) + Ranges().pushes(range2))
        self.assertEqual(result, output, '%s != %s' % (result, output))

    @ddt.data(
        (('sheet1!D7:F14',), ('C:E',)),
        (('sheet2!D7:F14',), ('sheet1!C4:E9',)),
        (("'[excel.xls]sheet2'!I7:L14",), ('H9:M12',)),
        )
    def test_invalid_add_ranges(self, case):
        range1, range2 = case
        with self.assertRaises(RangeValueError):
            Ranges().pushes(range1) + Ranges().pushes(range2)

    @ddt.data(
        ((('D7:F14',), ('C:E',)), '<Ranges>(D7:E14)'),
        ((('D7:F14',), ('C4:E9',)), '<Ranges>(D7:E9)'),
        ((('I7:L14',), ('H9:M12',)), '<Ranges>(I9:L12)'),
        ((('I7:L14',), ('J5:K16',)), '<Ranges>(J7:K14)'),
        ((('F24:I32',), ('G20:H26',)), '<Ranges>(G24:H26)'),
        ((('M23:P30',), ('L20:Q25',)), '<Ranges>(M23:P25)'),
        ((('M23:P30', 'L20:L25', 'Q20:Q25', 'M20:P22',), ('L20:Q25',)),
         '<Ranges>(M23:P25, L20:L25, Q20:Q25, M20:P22)'),
        ((('M23:P30', 'L20:L25', 'Q20:Q25', 'M20:P22',),
          ('P18:R27', 'K25:N32')),
         '<Ranges>(P23:P27, Q20:Q25, P20:P22, M25:N30, L25)'))
    def test_intersection_ranges(self, case):
        (range1, range2), result = case
        output = str(Ranges().pushes(range1) & Ranges().pushes(range2))
        self.assertEqual(result, output, '%s != %s' % (result, output))

    @ddt.data(
        (('D7:F14', 'C:C', 'D:E6', 'D15:E'),
         '<Ranges>(C:C, D:E15, F7:F14)'),
        (('D7:F14',), '<Ranges>(D7:F14)'),
        (('D7:D14', 'E7:E14', 'F7:F14'), '<Ranges>(D7:F14)')
    )
    def test_simplify_ranges(self, case):
        ranges, result = case
        output = str(Ranges().pushes(ranges).simplify())
        self.assertEqual(result, output, '%s != %s' % (result, output))

    @ddt.data(
        ((('B2:E5',), ('D4:F5',)),
         (([(1, 1, 2, 2),
            (1, 1, 2, 2),
            (3, 3, 4, 4),
            (3, 3, 4, 4)],),
          ([(4, 4, 1),
            (4, 4, 1)],)),
         [(4, 4),
          (4, 4)]),
        ((('B2:E5', 'G2:H5'), ('E2:G5',)),
         (([(1, 1, 2, 2),
            (1, 1, 2, 2),
            (3, 3, 4, 4),
            (3, 3, 4, 4)],
           [(5, 5),
            (5, 5),
            (5, 5),
            (5, 5)]),
          ([(2, 0, 5),
            (2, 0, 5),
            (4, 0, 5),
            (4, 0, 5)],)),
         [5, 5, 5, 5, 2, 2, 4, 4])
    )
    def test_value_intersection_ranges(self, case):
        (r1, r2), (v1, v2), result = case
        rng = Ranges().pushes(r1, v1) & Ranges().pushes(r2, v2)
        np.testing.assert_array_equal(result, rng.value)

    @ddt.data(
        ((('B2:E5',), ('D4:F5',)),
         (([(1, 1, 2, 2),
            (1, 1, 2, 2),
            (3, 3, 4, 4),
            (3, 3, 4, 4)],),
          (([(4, 4, 5),
             (4, 4, 5)],))),
         [5, 5, 1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4]),)
    def test_value_union_ranges(self, case):
        (r1, r2), (v1, v2), result = case
        rng = Ranges().pushes(r1, v1) | Ranges().pushes(r2, v2)
        output = rng.value
        np.testing.assert_array_equal(result, output)

    @ddt.data(
        ((('B2:E5',), ('D4:F5',)),
         (([(1, 1, 2, 2),
            (1, 1, 2, 2),
            (3, 3, 4, 4),
            (3, 3, 4, 4)],),
          (([(4, 4, 5),
             (4, 4, 5)],))),
         [(1, 1, 2, 2, ''),
          (1, 1, 2, 2, ''),
          (3, 3, 4, 4, 5),
          (3, 3, 4, 4, 5)]),)
    def test_value_add_ranges(self, case):
        (r1, r2), (v1, v2), result = case
        rng = Ranges().pushes(r1, v1) + Ranges().pushes(r2, v2)
        output = rng.value
        np.testing.assert_array_equal(np.asarray(result, object), output)

    def test_invalid_ranges(self):
        with self.assertRaises(ValueError):
            Ranges().push('reference')

    def test_empty_ranges_value(self):
        out = np.asarray([[Error.errors['#NULL!']]], object)
        np.testing.assert_array_equal(Ranges().value, out)
