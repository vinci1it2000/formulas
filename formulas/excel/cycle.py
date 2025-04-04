#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2025 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
A dependency-free version of networkx's implementation of `simple_cycles`.
"""

from collections import defaultdict


def _strong_connect(node, graph, counter, stack, result, lowlink, index):
    index[node] = lowlink[node] = counter[0]
    counter[0] += 1
    stack.append(node)

    for s in graph[node]:
        if s not in index:
            _strong_connect(s, graph, counter, stack, result, lowlink, index)
            lowlink[node] = min(lowlink[node], lowlink[s])
        elif s in stack:
            lowlink[node] = min(lowlink[node], index[s])

    if lowlink[node] == index[node]:
        connected_component = []

        while True:
            s = stack.pop()
            connected_component.append(s)
            if s == node:
                break
        result.append(connected_component[:])


def _strongly_connected_components(graph):
    # Tarjan's algorithm for finding SCC's
    # Robert Tarjan. "Depth-first search and linear graph algorithms."
    # SIAM journal on computing. 1972.
    # Code by Dries Verdegem, November 2012
    # Downloaded from http://www.logarithmic.net/pfh/blog/01208083168

    counter, stack, result, lowlink, index = [0], [], [], {}, {}
    for node in graph:
        if node not in index:
            _strong_connect(node, graph, counter, stack, result, lowlink, index)
    return result


def _remove_node(graph, target):
    # Completely remove a node from the graph
    # Expects values of G to be sets
    del graph[target]
    for nbrs in graph.values():
        nbrs.discard(target)


def _subgraph(graph, vertices):
    # Get the subgraph of G induced by set vertices
    # Expects values of G to be sets
    return {v: graph[v] & vertices for v in vertices}


def _unblock(thisnode, blocked, no_circuit):
    stack = {thisnode}
    while stack:
        node = stack.pop()
        if node in blocked:
            blocked.remove(node)
            stack.update(no_circuit[node])
            no_circuit[node].clear()


def simple_cycles(graph, copy=True, skip_nodes=()):
    # Yield every elementary cycle in python graph G exactly once
    # Expects a dictionary mapping from vertices to iterables of vertices

    if copy or skip_nodes:
        skip_nodes = set(skip_nodes)
        graph = {
            v: set(nbrs) - skip_nodes
            for v, nbrs in graph.items()
            if v not in skip_nodes
        }

    sccs = _strongly_connected_components(graph)
    while sccs:
        scc = sccs.pop()
        startnode = scc.pop()
        path, blocked, closed = [startnode], {startnode}, set()
        no_circuit = defaultdict(set)
        stack = [(startnode, list(graph[startnode]))]
        while stack:
            thisnode, nbrs = stack[-1]
            if nbrs:
                # noinspection PyUnresolvedReferences
                nextnode = nbrs.pop()
                if nextnode == startnode:
                    yield path[:]
                    closed.update(path)
                elif nextnode not in blocked:
                    path.append(nextnode)
                    stack.append((nextnode, list(graph[nextnode])))
                    closed.discard(nextnode)
                    blocked.add(nextnode)
                    continue
            if not nbrs:
                if thisnode in closed:
                    _unblock(thisnode, blocked, no_circuit)
                else:
                    for nbr in graph[thisnode]:
                        if thisnode not in no_circuit[nbr]:
                            no_circuit[nbr].add(thisnode)
                stack.pop()
                path.pop()
        _remove_node(graph, startnode)
        sccs.extend(_strongly_connected_components(_subgraph(graph, set(scc))))
