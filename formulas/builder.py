#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
It provides AstBuilder class.
"""

import collections
import schedula as sh
from .errors import FormulaError
from .tokens.operator import Operator
from .tokens.function import Function
from .tokens.operand import Operand
from .formulas import wrap_ranges_func
from .ranges import Ranges
from schedula.utils.alg import get_unused_node_id


class AstBuilder(collections.deque):
    def __init__(self, *args, dsp=None, nodes=None, match=None, **kwargs):
        super(AstBuilder, self).__init__(*args, **kwargs)
        self.match = match
        self.dsp = dsp or sh.Dispatcher(raises=True)
        self.nodes = nodes or {}
        self.missing_operands = set()

    def append(self, token):
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
                self.dsp.add_function(
                    function_id=get_id(dmap, token.name),
                    function=token.compile(),
                    inputs=inputs or None,
                    outputs=[out]
                )
            else:
                self.nodes[token] = n_id = get_id(dmap, out, 'c%d>{}')
                self.dsp.add_function(None, sh.bypass, [out], [n_id])
        elif isinstance(token, Operand):
            self.missing_operands.add(token)

        super(AstBuilder, self).append(token)

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

    def compile(self, references=None):
        dsp = self.dsp
        inputs = {}
        for k in set(dsp.data_nodes).intersection(references or {}):
            inputs[k] = Ranges().push(references[k])
        res, o = dsp(inputs), self.get_node_id(self[-1])
        dsp = dsp.get_sub_dsp_from_workflow(
            [o], graph=dsp.dmap, reverse=True, blockers=res,
            wildcard=False
        )
        dsp.nodes.update({k: v.copy() for k, v in dsp.nodes.items()})

        i = collections.OrderedDict()
        for k in sorted(dsp.data_nodes):
            if not dsp.dmap.pred[k]:
                if k in res:
                    v = res[k]
                    if isinstance(v, Ranges) and v.ranges:
                        i[k] = v
                    else:
                        dsp.add_data(data_id=k, default_value=v)
                else:
                    try:
                        i[k] = Ranges().push(k)
                    except ValueError:
                        i[k] = None

        dsp.nodes[o]['filters'] = wrap_ranges_func(sh.bypass),
        return sh.SubDispatchPipe(dsp, '=%s' % o, i, [o], wildcard=False)
