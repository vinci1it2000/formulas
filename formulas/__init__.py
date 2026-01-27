#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2025 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
It contains a comprehensive list of all modules and classes within formulas.

Modules:

.. currentmodule:: formulas

.. autosummary::
    :nosignatures:
    :toctree: toctree/formulas

    ~parser
    ~builder
    ~errors
    ~tokens
    ~functions
    ~ranges
    ~cell
    ~excel
"""

import functools
import os
import sys

from ._version import *  # noqa: F403

_all = {
    "ExcelModel": ".excel",
    "BOOK": ".excel",
    "SHEETS": ".excel",
    "CIRCULAR": ".excel",
    "ERR_CIRCULAR": ".excel",
    "Parser": ".parser",
    "get_functions": ".functions",
    "SUBMODULES": ".functions",
    "CELL": ".cell",
    "Ranges": ".ranges",
    "XlError": ".tokens.operand",
    "VALUE": ".tokens.operand",
    "REF": ".tokens.operand",
    "DIV": ".tokens.operand",
    "NA": ".tokens.operand",
    "NAME": ".tokens.operand",
    "NULL": ".tokens.operand",
    "NUM": ".tokens.operand",
}

__all__ = tuple(_all)


@functools.lru_cache(None)
def __dir__():
    return __all__ + (
        "__doc__",
        "__author__",
        "__updated__",
        "__title__",
        "__version__",
        "__license__",
        "__copyright__",
    )


def __getattr__(name):
    if name in _all:
        import importlib

        obj = getattr(importlib.import_module(_all[name], __name__), name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__} has no attribute {name}")


if sys.version_info[:2] < (3, 7) or os.environ.get("IMPORT_ALL") == "True":
    from .cell import CELL  # noqa: F401
    from .excel import BOOK, CIRCULAR, ERR_CIRCULAR, SHEETS, ExcelModel  # noqa: F401
    from .functions import SUBMODULES, get_functions  # noqa: F401
    from .parser import Parser  # noqa: F401
    from .ranges import Ranges  # noqa: F401
    from .tokens.operand import (  # noqa: F401
        DIV,
        NA,
        NAME,
        NULL,
        NUM,
        REF,
        VALUE,
        XlError,
    )
