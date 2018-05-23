#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

import unittest
import ddt
import schedula as sh
from formulas.cell import Cell


@ddt.ddt
class TestCell(unittest.TestCase):
    @ddt.data(
        ('A1', '=REF', {}, "<Ranges>(A1)=[[#REF!]]"),
        ('A1', '=(-INT(2))', {}, '<Ranges>(A1)=[[-2]]'),
        ('A1', '=(1+1)+(1+1)', {}, '<Ranges>(A1)=[[4]]'),
        ('A1', '=( 1 + 2 + 3)*(4 + 5)^(1/5)', {},
         '<Ranges>(A1)=[[9.311073443492159]]'),
        ('A1', '={1,2;1,2}', {}, '<Ranges>(A1)=[[1]]'),
        ('A1:B2', '={1,2;1,2}', {}, '<Ranges>(A1:B2)=[[1 2]\n [1 2]]'),
        ('A1', '=PI()', {}, '<Ranges>(A1)=[[3.141592653589793]]'),
        ('A1', '=INT(1)%+3', {}, '<Ranges>(A1)=[[3.01]]'),
        ('A1', '=SUM({1, 3; 4, 2})', {}, '<Ranges>(A1)=[[10]]'),
        ('A1', '=" "" a"', {}, '<Ranges>(A1)=[[\' " a\']]'),
        ('A1', '=#NULL!', {}, "<Ranges>(A1)=[[#NULL!]]"),
        ('A1', '=1 + 2', {}, '<Ranges>(A1)=[[3]]'),
        ('A1', '=AVERAGE(((123 + 4 + AVERAGE({1,2}))))', {},
         '<Ranges>(A1)=[[128.5]]'),
        ('A1', '="a" & "b"""', {}, '<Ranges>(A1)=[[\'ab"\']]'),
        ('A1', '=SUM(B2:B4)', {'B2:B4': ('', '', '')}, '<Ranges>(A1)=[[0]]'),
        ('A1', '=SUM(B2:B4)', {'B2:B4': ('', 1, '')}, '<Ranges>(A1)=[[1]]'),
        ('A1', '=MATCH("*b?u*",{"a",2.1,"ds  bau  dsd",4.1},0)', {},
         '<Ranges>(A1)=[[3]]'),
        ('A1', '=MATCH(4.1,{FALSE,2.1,TRUE,4.1},-1)', {},
         '<Ranges>(A1)=[[#N/A]]'),
        ('A1', '=HLOOKUP(-1.1,{-1.1,2.1,3.1,4.1;5,6,7,8},2,0)', {},
         '<Ranges>(A1)=[[5]]'),
        ('A1', '=HLOOKUP(-1.1,{-1.1,2.1,3.1,4.1;5,6,7,8},3,0)', {},
         '<Ranges>(A1)=[[#REF!]]'),
        ('A1', '=MATCH(1.1,{"b",4.1,"a",1.1})', {}, '<Ranges>(A1)=[[#N/A]]'),
        ('A1', '=MATCH(1.1,{4.1,2.1,3.1,1.1})', {}, '<Ranges>(A1)=[[#N/A]]'),
        ('A1', '=MATCH(4.1,{4.1,"b","a",1.1})', {}, '<Ranges>(A1)=[[4]]'),
        ('A1', '=MATCH(4.1,{"b",4.1,"a",1.1})', {}, '<Ranges>(A1)=[[2]]'),
        ('A1', '=MATCH(4.1,{4.1,"b","a",5.1},-1)', {}, '<Ranges>(A1)=[[1]]'),
        ('A1', '=MATCH(4.1,{"b",4.1,"a",5.1},-1)', {}, '<Ranges>(A1)=[[2]]'),
        ('A1', '=MATCH("b",{"b",4.1,"a",1.1})', {}, '<Ranges>(A1)=[[3]]'),
        ('A1', '=MATCH(3,{-1.1,2.1,3.1,4.1})', {}, '<Ranges>(A1)=[[2]]'),
        ('A1', '=MATCH(-1.1,{"b",4.1,"a",1.1})', {}, '<Ranges>(A1)=[[#N/A]]'),
        ('A1', '=MATCH(-1.1,{4.1,2.1,3.1,1.1},-1)', {}, '<Ranges>(A1)=[[4]]'),
        ('A1', '=MATCH(-1.1,{-1.1,2.1,3.1,4.1})', {}, '<Ranges>(A1)=[[1]]'),
        ('A1', '=MATCH(2.1,{4.1,2.1,3.1,1.1})', {}, '<Ranges>(A1)=[[2]]'),
        ('A1', '=MATCH(2.1,{4.1,2.1,3.1,1.1},-1)', {}, '<Ranges>(A1)=[[2]]'),
        ('A1', '=MATCH(2,{4.1,2.1,3.1,1.1},-1)', {}, '<Ranges>(A1)=[[3]]'),
        ('A1', '=LOOKUP(2.1,{4.1,2.1,3.1,1.1},{"L","ML","MR","R"})', {},
         '<Ranges>(A1)=[[\'ML\']]'),
        ('A1', '=LOOKUP("b",{"b",4.1,"a",1.1},{"L","ML","MR","R"})', {},
         '<Ranges>(A1)=[[\'MR\']]'),
        ('A1', '=LOOKUP(TRUE,{TRUE,4.1,FALSE,1.1},{"L","ML","MR","R"})', {},
         '<Ranges>(A1)=[[\'MR\']]'),
        ('A1', '=LOOKUP(4.1,{"b",4.1,"a",1.1},{"L","ML","MR","R"})', {},
         '<Ranges>(A1)=[[\'ML\']]'),
        ('A1', '=LOOKUP(2,{"b",4.1,"a",1.1},{"L","ML","MR","R"})', {},
         '<Ranges>(A1)=[[#N/A]]'),
        ('A1', '=LOOKUP(4.1,{4.1,2.1,3.1,1.1},{"L","ML","MR","R"})', {},
         '<Ranges>(A1)=[[\'R\']]'),
        ('A1', '=LOOKUP(4,{4.1,2.1,3.1,1.1},{"L","ML","MR","R"})', {},
         '<Ranges>(A1)=[[\'R\']]'),
        ('A1:D1', '=IF({0,-0.2,0},2,{1})', {},
         '<Ranges>(A1:D1)=[[1.0 2.0 1.0 #N/A]]'),
        # ('A1:D1', '=IF({0,-0.2,0},{2,3},{1})', {},
        #  '<Ranges>(A1:D1)=[[1 2 1 #N/A]]'),
        # ('A1:D1', '=IF({0,-2,0},{2,3},{1,4})', {},
        #  '<Ranges>(A1:D1)=[[1 2 #N/A #N/A]]')
    )
    def test_output(self, case):
        reference, formula, inputs, result = case
        dsp = sh.Dispatcher()
        cell = Cell(reference, formula).compile()
        assert cell.add(dsp)
        output = str(dsp(inputs)[cell.output])
        self.assertEqual(
            result, output,
            'Folmula({}): {} != {}'.format(formula, result, output)
        )

    @ddt.data(
        ('A1:D1', '=IF({0,-0.2,0},{2,3},{1})', {}),  # BroadcastError
        ('A1:D1', '=IF({0,-2,0},{2,3},{1,4})', {}),  # BroadcastError
    )
    def test_invalid(self, case):
        reference, formula, inputs = case
        with self.assertRaises(sh.DispatcherError):
            dsp = sh.Dispatcher()
            cell = Cell(reference, formula).compile()
            assert cell.add(dsp)
            dsp(inputs)
