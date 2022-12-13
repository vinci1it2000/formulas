#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2022 European Commission (JRC);
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
from .ranges import Ranges, _assemble_values, _shape, _get_indices_intersection
from .tokens.operand import Error, XlError, range2parts, _re_ref, _index2col

CELL = sh.Token('Cell')


class CellWrapper(sh.add_args):
    _args = 'func', 'parse_args', 'parse_kwargs'

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

    def __init__(self, reference, value, context=None, check_formula=True,
                 replace_missing_ref=True):
        self.func = self.range = self.inputs = self.output = None
        self.replace_missing_ref = replace_missing_ref
        if reference is not None:
            self.range = Ranges().push(reference, context=context)
            r = self.range.ranges[0]
            context = sh.combine_dicts(context or {}, base={
                'cr': r['r1'], 'cc': r['n1']
            })
            self.output = r['name']
        self.builder, self.value = None, sh.EMPTY
        prs = self.parser
        if check_formula and isinstance(value, str) and prs.is_formula(value):
            self.builder = prs.ast(value, context=context)[1]
        elif value is not None:
            self.value = value

    @property
    def __name__(self):
        if self.func:
            return self.func.__name__
        return self.output

    def compile(self, references=None, context=None):
        if not self.func and self.builder:
            func = self.builder.compile(
                references=references, context=context, **{CELL: self.range}
            )
            self.func = wrap_cell_func(func, self._args)
            self.update_inputs(references=references)
            self.builder = None
        return self

    def _missing_ref(self, inp, k):
        i = k
        if self.replace_missing_ref:
            m = _re_ref.match(k)
            i = m and m.groupdict()['excel_id'] and '#NAME?' or '#REF!'
            i = Error.errors[i]
        sh.get_nested_dicts(inp, i, default=list).append(k)

    def update_inputs(self, references=None):
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
            output, ctx = self.output, context or {}
            nodes.add(dsp.add_data(
                output, filters=self._output_filters(),
                default_value=self.value, initial_dist=sh.inf(1, 0)
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
                            dsp.add_data(k, val)
                        else:
                            try:
                                rng = Ranges.get_range(k, ctx)
                                f = functools.partial(format_output, rng),
                                dsp.add_data(k, filters=f)
                            except ValueError:
                                dsp.add_data(k)
                nodes.add(dsp.add_function(
                    self.__name__, self.func, inputs or None, [output]
                ))
        return nodes


class Ref(Cell):
    def __init__(self, reference, value, context=None, check_formula=True):
        context = (context or {}).copy()
        context.update({
            k: v for k, v in _re_ref.match(reference).groupdict().items()
            if v is not None
        })
        ref = context.pop('ref')
        super(Ref, self).__init__(None, value, context, check_formula)
        self.output = range2parts(['name'], ref=ref, **context)['name']

    def _missing_ref(self, inp, k):
        sh.get_nested_dicts(inp, k, default=list).append(k)

    def _output_filters(self):
        return ()

    def compile(self, references=None, context=None):
        super(Ref, self).compile(references=references, context=context)
        if self.inputs:
            self.func.dsp.nodes[self.func.outputs[0]].pop('filters', None)
        return self


class RangesAssembler:
    @staticmethod
    def _range_indices(rng):
        return {
            (j, i)
            for r in rng.ranges
            for i in range(int(r['r1']), int(r['r2']) + 1)
            for j in range(r['n1'], r['n2'] + 1)
        }

    def __init__(self, ref, context=None, compact=1):
        self.range = Ranges().push(ref, context=context)
        self.missing = self._range_indices(self.range)
        self.inputs = collections.OrderedDict()
        self.compact = compact or 1

    @property
    def output(self):
        return self.range.ranges[0]['name']

    def push(self, indices, output=None):
        it = {i for i in self.missing if i in indices}
        if it:
            self.missing.difference_update(it)
            it = (indices[i] for i in it) if output is None else (output,)
            self.inputs.update(dict.fromkeys(it, None))
        return self.missing

    def add(self, dsp):
        base = self.range.ranges[0]
        sheet_id = base['sheet_id']
        ists = {}
        nodes = dsp.default_values
        _name = f'{sheet_id}!%s' if sheet_id else '%s'
        for n, r in tuple(self.missing):
            c = _index2col(n)
            ref = '{}{}'.format(c, r)
            name = _name % ref
            ist = {
                'r1': r, 'r2': r, 'c1': c, 'c2': c, 'n1': n, 'n2': n,
                'ref': ref, 'name': name, 'sheet_id': sheet_id
            }
            if name in nodes:
                self.inputs[name] = _get_indices_intersection(base, ist)
                self.missing.remove((n, r))
            else:
                ists[name] = ist

        if len(ists) <= self.compact:
            for k, ist in ists.items():
                self.inputs[k] = _get_indices_intersection(base, ist)
                f = functools.partial(format_output, ist),
                dsp.add_data(k, [[sh.EMPTY]], filters=f)
        else:
            if sh.SELF not in nodes:
                dsp.add_data(sh.SELF, sh.inf(2, 0))
            self.inputs[sh.SELF] = ists
        if list(self.inputs) != [self.output]:
            dsp.add_function(None, self, self.inputs or None, [self.output])

    @property
    def __name__(self):
        return '=%s' % self.output

    def __call__(self, *cells):
        base = self.range.ranges[0]
        if sh.SELF in self.inputs:
            out = np.empty(_shape(**base), object)
            out[:] = sh.EMPTY
            ists = self.inputs[sh.SELF]
            sol = cells[-1].solution
            cells = cells[:-1]
            for n, v in ists.items():
                if n in sol:
                    if isinstance(v, dict):
                        v = ists[n] = _get_indices_intersection(base, v)
                    i, j = v
                    v = sol[n]
                    if isinstance(sol[n], Ranges):
                        v = v.value
                    out[i, j] = v
        else:
            out = np.empty(_shape(**base), object)
        for c, ind in zip(cells, self.inputs.values()):
            if ind:
                out[ind[0], ind[1]] = c.value
            else:
                _assemble_values(base, c.values, out)
        return out
