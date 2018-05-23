#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

import unittest
import ddt
import inspect
import numpy as np
import schedula as sh
from formulas.parser import Parser
from formulas.errors import FormulaError
from formulas.ranges import Ranges
from formulas.formulas import wrap_ranges_func


@ddt.ddt
class TestParser(unittest.TestCase):
    @ddt.data(
        ('=[0]S.H.E.E.t!_r_e_f_1__', '[0]S.H.E.E.T!_R_E_F_1__'),
        ('=INDIRECT("S.H.E.E.t!_r_e_f_1__")', 'S.H.E.E.T!_R_E_F_1__'),
        ('=INDIRECT("_sheet1.disco!ref")', '_SHEET1.DISCO!REF'),
        ('=L4:N15 : J5:L12', 'L4:N15:J5:L12'),
        ('=L4:N15 : c', 'L4:N15:C'),
        ('=ciao:bau', 'CIAO:BAU'),
        ('=SUM(L4:N15 (J5:L12, J5:L12, N5:P12, J5:L12, J5:L12))',
         'SUM(L4:N15 (J5:L12,J5:L12,N5:P12,J5:L12,J5:L12))'),
        ('=SUM(L4:N15 (J5:L12, N5:P12))', 'SUM(L4:N15 (J5:L12,N5:P12))'),
        ('=(-INT(2))', '(u-INT(2))'),
        ('=(1+1)+(1+1)', '(1+1)+(1+1)'),
        ('=( 1 + 2 + 3)*(4 + 5)^(1/5)', '(1+2+3)*(4+5)^(1/5)'),
        ('={1,2;1,2}', 'ARRAY(ARRAY(1,2)ARRAY(1,2))'),
        ('=PI()', 'PI()'),
        ('=INT(1)%+3', 'INT(1)%+3'),
        ('=INT(1)%-3', 'INT(1)%-3'),
        ('= 1 + 2 + 3 * 4 ^ 5 * SUM(a, 1+b)', '1+2+3*4^5*SUM(A,1+B)'),
        ('=SUM({1})', 'SUM(ARRAY(ARRAY(1)))'),
        ('= a b', 'A B'),
        ('="  a"', '  a'),
        ('=#NULL!', '#NULL!'),
        ('=1 + 2', '1+2'),
        ('{=a1:b1 + INDIRECT("A1:B2")}', 'A1:B1+A1:B2'),
        ('= a + IF(a, b, c)', 'A+IF(A,B,C)'),
        ('=AVERAGE(((123 + 4 + AVERAGE(A1:A2))))',
         'AVERAGE(((123+4+AVERAGE(A1:A2))))'),
        ('=a,b', 'A,B'),
        ('=a b', 'A B'),
        ('=MYFORMULA(1)', 'MYFORMULA(1)'),
        ('=IF(G9:G12<>H10:H13,1,0)', 'IF(G9:G12<>H10:H13,1,0)'),
        ('=SUM(a,b,d,e,f,g,Sheet2!B3:B4)', 'SUM(A,B,D,E,F,G,SHEET2!B3:B4)'),
        ('=1000/(Sheet1!B1*Sheet1!B2)', '1000/(SHEET1!B1*SHEET1!B2)'),
        ('=10  ^  -  2', '10^u-2'),
        ('=10^- + -  + + +2', '10^u+2'),
        ('=ATAN2( 10 , 2)', 'ATAN2(10,2)'),
        ('=DAYS360( 10 , 2)', 'DAYS360(10,2)'),
    )
    def test_valid_formula(self, case):
        inputs, result = case
        tokens, ast = Parser().ast(inputs)
        output = ''.join(t.name for t in tokens)
        self.assertEqual(result, output, '%s != %s' % (result, output))

    @ddt.data(
        '=-', '={}', '=1  -*  4', '={1;2,2}', '= a + IF((a, b, c)',
        '= a + IF(}a, b, c)', '1+2', '=a 1', '=a INT(1)', '=1)', '=Sheet1!')
    def test_invalid_formulas(self, inputs):
        with self.assertRaises(FormulaError):
            Parser().ast(inputs)

    @ddt.data(
        ({}, '=(L4:N7 (K5:L6, N5:O6))', (
            Ranges().pushes(('L5:L6', 'N5:N6'), ([2, 2], [3, 3])),
        ), '[3 3 2 2]'),
        ({}, '=(a (b, c))', (
            Ranges().push('L4:N6', [(1, 1, 1), (2, 1, 3), (2, 1, 3)]),
            Ranges().push('K5:L6', [(2, 2), (2, 2)]),
            Ranges().push('N5:O6', [(3, 3), (3, 3)])
        ),
         '[3 3 2 2]'),
        ({'A': 'L4:N7', 'B': 'K5:L6', 'C': 'N5:O6'}, '=(a (b, c))', (
            Ranges().pushes(('L5:L6', 'N5:N6'), ([2, 2], [3, 3])),
        ),
         '[3 3 2 2]'),
        ({}, '=(-INT(2))', (), '-2'),
        ({}, '=(1+1)+(1+1)', (), '4'),
        ({}, '=( 1 + 2 + 3)*(4 + 5)^(1/5)', (), '9.311073443492159'),
        ({}, '={1,2;1,2}', (), '[[1 2]\n [1 2]]'),
        ({}, '=PI()', (), '3.141592653589793'),
        ({}, '=INT(1)%+3', (), '3.01'),
        ({}, '=SUM({1, 3; 4, 2})', (), '10'),
        ({}, '=" "" a"', (), ' " a'),
        ({}, '=#NULL!',  (), '#NULL!'),
        ({}, '=1 + 2', (), '3'),
        ({}, '=AVERAGE(((123 + 4 + AVERAGE({1,2}))))', (), '128.5'),
        ({}, '="a" & "b"""', (), 'ab"'),
        ({}, '=-2', (), '-2'),
        ({}, '=10*+2 + 10^--2 + 10/-2', (), '115.0'),
        ({}, '=10>+2', (), 'True'),
        ({}, '=10=+10', (), 'True'),
        ({}, '=ISERR(#VALUE!)', (), 'True'))
    def test_compile(self, case):
        references, formula, inputs, result = case
        func = Parser().ast(formula)[1].compile(references)
        output = str(func(*inputs))
        self.assertEqual(result, output, '{} != {}'.format(result, output))

    @ddt.data(
        ('=MYFORMULA(1)', ())
    )
    def test_invalid_compile(self, case):
        formula, inputs = case
        with self.assertRaises(sh.DispatcherError):
            Parser().ast(formula)[1].compile()(*inputs)

    def test_ast_function(self):
        def _func(a, b):
            """Doc."""
            return a + b

        func = wrap_ranges_func(_func)
        self.assertEqual(func.__name__, _func.__name__)
        self.assertEqual(func.__doc__, _func.__doc__)
        self.assertEqual(inspect.signature(func), inspect.signature(_func))

        rng1 = Ranges().push('A1:A1', [[1]])
        output = func(rng1, Ranges().push('B1:B1'))
        self.assertEqual(output, sh.NONE)

        output = func(rng1, Ranges().push('B1:B1', [[2]]))
        np.testing.assert_array_equal([[3]], output)
