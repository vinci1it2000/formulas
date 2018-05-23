#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
It provides Cell class.
"""
import collections
import functools
import numpy as np
import schedula as sh
from .parser import Parser
from .ranges import Ranges, _assemble_values
from .tokens.operand import Error, XlError


def wrap_cell_func(func, parse_args=lambda *a: a, parse_kwargs=lambda **kw: kw):
    def wrapper(*args, **kwargs):
        return func(*parse_args(*args), **parse_kwargs(**kwargs))
    return functools.update_wrapper(wrapper, func)


def format_output(*args, **kwargs):
    return Ranges().push(*args, **kwargs)


class Cell(object):
    parser = Parser()

    def __init__(self, reference, value, context=None):
        self.func = None
        self.inputs = None
        self.range = Ranges().push(reference, context=context)
        self.value = sh.EMPTY
        self.tokens, self.builder = (), None
        if isinstance(value, str) and self.parser.is_formula(value):
            self.tokens, self.builder = self.parser.ast(value, context=context)
        elif value is not None:
            self.value = value

    @property
    def __name__(self):
        if self.func:
            return self.func.__name__
        return self.output

    @property
    def output(self):
        return self.range.ranges[0]['name']

    def compile(self, references=None):
        if self.builder:
            func = self.builder.compile(references=references)
            self.func = wrap_cell_func(func, self._args)
            self.update_inputs(references=references)
        return self

    def update_inputs(self, references=None):
        if not self.builder:
            return
        self.inputs = inp = collections.OrderedDict()
        for k, rng in self.func.inputs.items():
            try:
                rng = rng or Ranges().push((references or {})[k])
            except KeyError:
                sh.get_nested_dicts(
                    inp, Error.errors['#REF!'], default=list
                ).append(k)
                continue
            for r in rng.ranges:
                sh.get_nested_dicts(inp, r['name'], default=list).append(k)

    def _args(self, *args):
        assert len(args) == len(self.inputs)
        i = {}
        for links, v in zip(self.inputs.values(), args):
            for k in links:
                i[k] = (v + i[k]) if k in i else v
        return sh.selector(self.func.inputs, i, output_type='list')

    def add(self, dsp, context=None):
        if self.func or self.value is not sh.EMPTY:
            directory = context and context.get('directory') or '.'
            output = self.output
            f = functools.partial(format_output, output, context=context)
            dsp.add_data(output, filters=(f,), default_value=self.value,
                         directory=directory)

            if self.func:
                inputs = self.inputs
                for k in inputs or ():
                    if k not in dsp.nodes:
                        if isinstance(k, XlError):
                            val = Ranges().push(
                                'A1:', np.asarray([[k]], object)
                            )
                            dsp.add_data(k, val, directory=directory)
                        else:
                            f = functools.partial(format_output, k,
                                                  context=context)
                            dsp.add_data(k, filters=(f,), directory=directory)

                dsp.add_function(
                    self.__name__, self.func, inputs or None, [output]
                )
            return True


class RangesAssembler(object):
    def __init__(self, ref, context=None):
        self.missing = self.range = Ranges().push(ref, context=context)
        self.inputs = []

    @property
    def output(self):
        return self.range.ranges[0]['name']

    def push(self, cell):
        if self.missing.ranges and (self.missing & cell.range).ranges:
            self.missing = self.range - cell.range
            self.inputs.append(cell.output)

    @property
    def __name__(self):
        return '=%s' % self.output

    def __call__(self, *cells):
        base = self.range.ranges[0]
        values = sh.combine_dicts(*(c.values for c in cells))
        return _assemble_values(base, values, sh.EMPTY)
