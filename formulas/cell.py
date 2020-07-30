#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2020 European Commission (JRC);
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
from .tokens.operand import Error, XlError, range2parts
from .ranges import Ranges, _assemble_values, _shape, _get_indices_intersection

CELL = sh.Token('Cell')


class CellWrapper(sh.add_args):
    def __init__(self, func, parse_args, parse_kwargs):
        super(CellWrapper, self).__init__(func, n=0)
        self.parse_args = parse_args
        self.parse_kwargs = parse_kwargs

    def __call__(self, *args, **kwargs):
        try:
            return self.func(
                *self.parse_args(*args), **self.parse_kwargs(**kwargs)
            )
        except sh.DispatcherError as ex:
            if isinstance(ex.ex, NotImplementedError):
                return Error.errors['#NAME?']
            raise ex

    def check_cycles(self, cycle):
        from .excel.cycle import simple_cycles
        fn, k, cells = self.func, 'solve_cycle', set()
        f_nodes, o, inputs = fn.dsp.function_nodes, fn.outputs[0], fn.inputs
        dmap = {v: set(nbrs) for v, nbrs in fn.dsp.dmap.succ.items()}
        dmap[o] = set(cycle).intersection(inputs)
        for c in map(set, simple_cycles(dmap, False)):
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

    def __init__(self, reference, value, context=None, check_formula=True):
        self.func = self.range = self.inputs = self.output = None
        if reference is not None:
            self.range = Ranges().push(reference, context=context)
            self.output = self.range.ranges[0]['name']
        self.tokens, self.builder, self.value = (), None, sh.EMPTY
        prs = self.parser
        if check_formula and isinstance(value, str) and prs.is_formula(value):
            self.tokens, self.builder = prs.ast(value, context=context)
        elif value is not None:
            self.value = value

    @property
    def __name__(self):
        if self.func:
            return self.func.__name__
        return self.output

    def compile(self, references=None):
        if self.builder:
            func = self.builder.compile(
                references=references, **{CELL: self.range}
            )
            self.func = wrap_cell_func(func, self._args)
            self.update_inputs(references=references)
        return self

    def _missing_ref(self, inp, k):
        sh.get_nested_dicts(inp, Error.errors['#REF!'], default=list).append(k)

    def update_inputs(self, references=None):
        if not self.builder:
            return
        self.inputs = inp = collections.OrderedDict()
        references, get = references or set(), sh.get_nested_dicts
        for k, rng in self.func.inputs.items():
            if k in references:
                get(inp, k, default=list).append(k)
            else:
                try:
                    for r in rng.ranges:
                        get(inp, r['name'], default=list).append(k)
                except AttributeError:
                    self._missing_ref(inp, k)

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

    def _output_filters(self):
        return functools.partial(format_output, self.range.ranges[0]),

    def add(self, dsp, context=None):
        nodes = set()
        if self.func or self.value is not sh.EMPTY:
            directory = context and context.get('directory') or '.'
            output = self.output
            nodes.add(dsp.add_data(
                output, filters=self._output_filters(),
                default_value=self.value, directory=directory
            ))
            if self.func:
                inputs = self.inputs
                nodes.update(inputs)
                for k in inputs or ():
                    if k not in dsp.nodes:
                        if isinstance(k, XlError):
                            val = Ranges().push(
                                'A1:', np.asarray([[k]], object)
                            )
                            dsp.add_data(k, val, directory=directory)
                        else:
                            try:
                                rng = Ranges.get_range(
                                    Ranges.format_range, k, context
                                )
                                f = functools.partial(format_output, rng),
                            except ValueError:
                                f = ()
                            dsp.add_data(k, filters=f, directory=directory)
                nodes.add(dsp.add_function(
                    self.__name__, self.func, inputs or None, [output]
                ))
        return nodes


class Ref(Cell):
    def __init__(self, reference, value, context=None, check_formula=True):
        context = context or {}
        super(Ref, self).__init__(None, value, context, check_formula)
        self.output = range2parts(None, ref=reference, **context)['name']

    def _missing_ref(self, inp, k):
        sh.get_nested_dicts(inp, k, default=list).append(k)

    def _output_filters(self):
        return ()

    def compile(self, references=None):
        super(Ref, self).compile()
        if self.inputs:
            self.func.dsp.nodes[self.func.outputs[0]].pop('filters', None)
        elif self.value is not sh.EMPTY:
            self.value, self.func = self.func(), None
        return self


class RangesAssembler:
    def __init__(self, ref, context=None):
        self.missing = self.range = Ranges().push(ref, context=context)
        self.inputs = collections.OrderedDict()

    @property
    def output(self):
        return self.range.ranges[0]['name']

    def push(self, cell):
        for b in self.missing.intersect(cell.range):
            if b:
                self.missing = self.missing - cell.range
                self.inputs[cell.output] = None
                break

    def add(self, dsp):
        base = self.range.ranges[0]
        for rng in self.missing.ranges:
            ctx = sh.selector(('excel', 'sheet'), rng)
            for r in range(int(rng['r1']), int(rng['r2']) + 1):
                for n in range(rng['n1'], rng['n2'] + 1):
                    ist = Ranges.format_range(
                        ('name', 'n1', 'n2'), n1=n, r1=r, **ctx
                    )
                    k = ist['name']
                    self.inputs[k] = _get_indices_intersection(base, ist)
                    f = functools.partial(format_output, ist),
                    dsp.add_data(k, [[sh.EMPTY]], filters=f)
        if list(self.inputs) != [self.output]:
            dsp.add_function(None, self, self.inputs or None, [self.output])

    @property
    def __name__(self):
        return '=%s' % self.output

    def __call__(self, *cells):
        base = self.range.ranges[0]
        out = np.empty(_shape(**base), object)
        for c, ind in zip(cells, self.inputs.values()):
            if ind:
                out[ind[0], ind[1]] = c.value
            else:
                _assemble_values(base, c.values, out)
        return out
