#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
It contains a comprehensive list of all modules and classes within formulas.

Modules:

.. currentmodule:: formulas

.. autosummary::
    :nosignatures:
    :toctree: _build/formulas

    ~parser
    ~builder
    ~errors
    ~tokens
    ~functions
    ~ranges
    ~cell
    ~excel
"""
import os
import sys
import functools
from ._version import *

_all = {
    'ExcelModel': '.excel',
    'Parser': '.parser',
    'get_functions': '.functions',
    'SUBMODULES': '.functions',
    'CELL': '.cell',
    'Ranges': '.ranges'
}

__all__ = tuple(_all)


@functools.lru_cache(None)
def __dir__():
    return __all__ + (
        '__doc__', '__author__', '__updated__', '__title__', '__version__',
        '__license__', '__copyright__'
    )


def __getattr__(name):
    if name in _all:
        import importlib
        obj = getattr(importlib.import_module(_all[name], __name__), name)
        globals()[name] = obj
        return obj
    raise AttributeError("module %s has no attribute %s" % (__name__, name))


if sys.version_info[:2] < (3, 7) or os.environ.get('IMPORT_ALL') == 'True':
    # noinspection PyUnresolvedReferences
    from .excel import ExcelModel
    # noinspection PyUnresolvedReferences
    from .parser import Parser
    # noinspection PyUnresolvedReferences
    from .functions import get_functions, SUBMODULES
    # noinspection PyUnresolvedReferences
    from .cell import CELL
    # noinspection PyUnresolvedReferences
    from .ranges import Ranges
