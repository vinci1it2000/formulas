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
import copy
import collections
import functools
import numpy as np
import schedula as sh
from .parser import Parser
from .ranges import Ranges, _assemble_values
from .tokens.operand import Error, XlError

CELL = sh.Token('Cell')


class CellWrapper:
    def __init__(self, func, parse_args, parse_kwargs):
        self.func = func
        self.parse_args = parse_args
        self.parse_kwargs = parse_kwargs

    def __call__(self, *args, **kwargs):
        return self.func(*self.parse_args(*args), **self.parse_kwargs(**kwargs))

    def check_cycles(self, cycle):
        import networkx as nx
        func = self.func
        f_nodes, o, cells = func.dsp.function_nodes, func.outputs[0], set()
        dmap, inputs, k = func.dsp.dmap.copy(), func.inputs, 'solve_cycle'
        dmap.add_edges_from((o, i) for i in inputs if i in cycle)
        for c in map(set, nx.simple_cycles(dmap)):
            for n in map(f_nodes.get, c.intersection(f_nodes)):
                if k in n and n[k](*(i in c for i in n['inputs'])):
                    cells.update(c.intersection(inputs))
                    break
            else:
                return set()
        return cells


def wrap_cell_func(func, parse_args=lambda *a: a, parse_kwargs=lambda **kw: kw):
    wrapper = CellWrapper(func, parse_args, parse_kwargs)
    return functools.update_wrapper(wrapper, func)


def format_output(rng, value):
    return Ranges().set_value(rng, value)


class Cell:
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
            func = self.builder.compile(
                references=references, **{CELL: self.range}
            )
            self.func = wrap_cell_func(func, self._args)
            self.update_inputs(references=references)
        return self

    def update_inputs(self, references=None):
        if not self.builder:
            return
        self.inputs = inp = collections.OrderedDict()
        ref = references or {}
        for k, rng in self.func.inputs.items():
            try:
                rng = rng or Ranges().push(ref[k])
            except KeyError:
                sh.get_nested_dicts(
                    inp, Error.errors['#REF!'], default=list
                ).append(k)
                continue
            for r in rng.ranges:
                sh.get_nested_dicts(inp, r['name'], default=list).append(k)

    def _args(self, *args):
        assert len(args) == len(self.inputs)
        inputs = copy.deepcopy(self.func.inputs)
        for links, v in zip(self.inputs.values(), args):
            for k in links:
                try:
                    inputs[k].values.update(v.values)
                except AttributeError:  # Reference.
                    inputs[k] = v
        return inputs.values()

    def add(self, dsp, context=None):
        if self.func or self.value is not sh.EMPTY:
            directory = context and context.get('directory') or '.'
            output = self.output
            rng = Ranges.get_range(Ranges.format_range, output, context)
            f = functools.partial(format_output, rng)
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
                            rng = Ranges.get_range(
                                Ranges.format_range, k, context
                            )
                            f = functools.partial(format_output, rng)
                            dsp.add_data(k, filters=(f,), directory=directory)

                dsp.add_function(
                    self.__name__, self.func, inputs or None, [output]
                )
            return True


class RangesAssembler:
    def __init__(self, ref, context=None):
        self.missing = self.range = Ranges().push(ref, context=context)
        self.inputs = []

    @property
    def output(self):
        return self.range.ranges[0]['name']

    def push(self, cell):
        if self.missing.ranges and any(self.missing.intersect(cell.range)):
            self.missing = self.range - cell.range
            self.inputs.append(cell.output)

    @property
    def __name__(self):
        return '=%s' % self.output

    def __call__(self, *cells):
        base = self.range.ranges[0]
        values = {}
        for c in cells:
            values.update(c.values)
        return _assemble_values(base, values, sh.EMPTY)
