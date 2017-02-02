#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
It provides Cell class.
"""
import collections
import functools
import schedula.utils as sh_utl
from .parser import Parser
from .ranges import Ranges
from .tokens.operand import Error


class Cell(object):
    parser = Parser()

    def __init__(self, reference, formula, context=None):
        self.func = None
        self.inputs = None
        self.range = Ranges().push(reference, context=context)
        self.tokens, self.builder = self.parser.ast(formula, context=context)

    @property
    def __name__(self):
        return self.func.__name__

    @property
    def output(self):
        return self.range.ranges[0]['name']

    def compile(self, references=None):
        self.func = self.builder.compile(references=references)
        self.update_inputs(references=references)
        return self

    def update_inputs(self, references=None):
        self.inputs = inp = collections.OrderedDict()
        for k, rng in self.func.inputs.items():
            try:
                rng = rng or Ranges().push((references or {})[k])
            except KeyError:
                self.inputs = None
                break
            for r in rng.ranges:
                sh_utl.get_nested_dicts(inp, r['name'], default=list).append(k)

    def _args(self, *args):
        assert len(args) == len(self.inputs)
        i = {}
        for links, v in zip(self.inputs.values(), args):
            for k in links:
                i[k] = (v + i[k]) if k in i else v
        return sh_utl.selector(self.func.inputs, i, output_type='list')

    def __call__(self, *args):
        if self.inputs is None:
            return Error.errors['#REF!']
        return self.func(*self._args(*args))

    def add(self, dsp, context=None):
        inputs, outputs = self.inputs, [self.output]
        for k in list(inputs or []) + outputs:
            if k not in dsp.nodes:
                f = functools.partial(Ranges().push, k, context=context)
                dsp.add_data(k, filters=(f,))
        dsp.add_function(self.__name__, self, inputs or None, outputs)
        return dsp
