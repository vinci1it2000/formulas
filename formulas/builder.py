#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
It provides AstBuilder class.
"""
import typing as t
import functools
import collections
import schedula as sh
from .errors import FormulaError, RangeValueError, InvalidRangeError
from .tokens import Token
from .tokens.operator import Operator
from .tokens.function import Function
from .tokens.operand import Operand
from .functions import wrap_ranges_func, COMPILING
from .ranges import Ranges
from schedula.utils.utl import get_unused_node_id


@functools.lru_cache(None)
def _default_filter():
    return wrap_ranges_func(sh.bypass),


class AstBuilder:
    compile_class = sh.DispatchPipe

    def __init__(self, dsp: t.Optional[sh.Dispatcher] = None, nodes=None, match: t.Optional[t.Dict[str, t.Any]] = None):
        self._deque: t.Deque = collections.deque()
        self.match = match
        self.dsp = dsp or sh.Dispatcher(
            raises=lambda e: not isinstance(e, (
                NotImplementedError, RangeValueError, InvalidRangeError
            ))
        )
        self.nodes = nodes or {}
        self.missing_operands: t.Set[t.Any] = set()

    def __len__(self):
        return len(self._deque)

    def __getitem__(self, index: int):
        return self._deque[index]

    def pop(self):
        return self._deque.pop()

    def append(self, token: t.Type[Token]):
        if isinstance(token, (Operator, Function)):
            try:
                tokens = [self.pop() for _ in range(token.get_n_args)][::-1]
            except IndexError:
                raise FormulaError()
            token.update_input_tokens(*tokens)
            inputs = [self.get_node_id(i) for i in tokens]
            token.set_expr(*tokens)
            out, dmap, get_id = token.node_id, self.dsp.dmap, get_unused_node_id
            if out not in self.dsp.nodes:
                func = token.compile()
                kw = {
                    'function_id': get_id(dmap, token.name),
                    'function': func,
                    'inputs': inputs or None,
                    'outputs': [out]
                }
                if isinstance(func, dict):
                    _inputs = func.get('extra_inputs', {})
                    for k, v in _inputs.items():
                        if v is not sh.NONE:
                            self.dsp.add_data(k, v)
                    kw['inputs'] = (list(_inputs) + inputs) or None
                    kw.update(func)
                self.dsp.add_function(**kw)
            else:
                self.nodes[token] = n_id = get_id(dmap, out, 'c%d>{}')
                self.dsp.add_function(None, sh.bypass, [out], [n_id])
        elif isinstance(token, Operand):
            self.missing_operands.add(token)
        
        
        self._deque.append(token)

    def get_node_id(self, token):
        if token in self.nodes:
            return self.nodes[token]
        if isinstance(token, Operand):
            self.missing_operands.remove(token)
            token.set_expr()
            kw = {}
            if not token.attr.get('is_reference', False):
                kw['default_value'] = token.compile()
            node_id = self.dsp.add_data(data_id=token.node_id, **kw)
        else:
            node_id = token.node_id
        self.nodes[token] = node_id
        return node_id

    def finish(self):
        for token in list(self.missing_operands):
            self.get_node_id(token)

    def compile(self, references=None, context=None, **inputs) -> t.Callable[..., t.Any]:
        """Compiles a given expression into a Schedula DispatchPipe, which itself acts as a Callable.

        Args:
            references (_type_, optional): _description_. Defaults to None.
            context (_type_, optional): _description_. Defaults to None.

        Returns:
            t.Callable[..., t.Any]: The Schedula DispatchPipe (in practice a Callable)
        """        
        dsp, inp = self.dsp, inputs.copy()
        for k, ref in (references or {}).items():
            if k in dsp.data_nodes:
                if isinstance(ref, Ranges):
                    inp[k] = ref
                elif ref is not None:
                    inp[k] = Ranges().push(ref)
        inp[COMPILING] = True
        res, o = dsp(inp), self.get_node_id(self[-1])
        dsp = dsp.get_sub_dsp_from_workflow(
            [o], graph=dsp.dmap, reverse=True, blockers=res,
            wildcard=False
        )
        res[COMPILING] = False
        dsp.nodes.update({k: v.copy() for k, v in dsp.nodes.items()})

        i = collections.OrderedDict()
        for k in sorted(dsp.data_nodes):
            if not dsp.dmap.pred[k]:
                if k in res:
                    v = res[k]
                    if k not in inputs and isinstance(v, Ranges) and v.ranges:
                        i[k] = v
                    else:
                        dsp.add_data(data_id=k, default_value=v)
                else:
                    try:
                        i[k] = Ranges().push(k, context=context)
                    except ValueError:
                        i[k] = None
        dsp.raises = True
        dsp.nodes[o]['filters'] = _default_filter()
        return self.compile_class(
            dsp, '=%s' % o, i, [o], wildcard=False, shrink=False
        )
