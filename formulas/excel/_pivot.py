# !/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2026 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
Best-effort discovery of pivot-table source ranges.

This is a deliberately small helper used by :func:`GETPIVOTDATA` so that a
reference *inside* a pivot table can be resolved to the underlying data range
that feeds the pivot.  We do not attempt to reproduce pivot semantics (grand
totals, calculated fields, grouping, etc.); we only locate the source data so
that the caller can aggregate it directly.

If ``openpyxl`` does not expose pivot metadata for the workbook -- or if the
range is not inside any pivot table -- :func:`find_pivot_source` returns
``None``.
"""

from ..ranges import Ranges


def _iter_pivot_tables(workbook):
    """Yield ``(worksheet, pivot)`` pairs for every pivot we can reach."""
    for ws in getattr(workbook, 'worksheets', []) or []:
        for pivot in getattr(ws, '_pivots', None) or []:
            yield ws, pivot


def _pivot_location_ref(pivot):
    """Return the ``A1:B2``-style location of a pivot table, or ``None``."""
    try:
        location = pivot.location
    except AttributeError:
        return None
    ref = getattr(location, 'ref', None) or getattr(pivot, 'ref', None)
    return ref


def _pivot_source_ref(pivot):
    """Return ``(sheet, A1)`` for the pivot's source data, or ``None``."""
    cache = getattr(pivot, 'cache', None)
    if cache is None:
        return None
    source = getattr(cache, 'cacheSource', None) or getattr(cache, 'source', None)
    if source is None:
        return None
    worksheet_source = getattr(source, 'worksheetSource', None)
    if worksheet_source is None:
        return None
    ref = getattr(worksheet_source, 'ref', None)
    sheet = getattr(worksheet_source, 'sheet', None)
    if ref is None or sheet is None:
        return None
    return sheet, ref


def _ranges_contains(outer_sheet, outer_ref, target):
    """Return True if a Ranges ``target`` lies inside ``outer_sheet!outer_ref``."""
    try:
        outer = Ranges().push('{}!{}'.format(outer_sheet, outer_ref))
    except Exception:
        return False
    try:
        intersection = outer & target
    except Exception:
        return False
    return bool(getattr(intersection, 'ranges', ()))


def find_pivot_source(model, ref):
    """Map a pivot-table reference to its underlying source data Ranges.

    Parameters
    ----------
    model : formulas.excel.ExcelModel
        Used only to discover the workbook file paths; if the model has no
        loaded books we silently return ``None``.
    ref : formulas.ranges.Ranges
        The candidate reference to resolve.

    Returns
    -------
    formulas.ranges.Ranges | None
        A new ``Ranges`` covering the pivot's source data when ``ref`` falls
        inside a known pivot table; otherwise ``None``.
    """
    if not isinstance(ref, Ranges) or not ref.ranges:
        return None

    try:
        import openpyxl
    except ImportError:
        return None

    book_paths = []
    try:
        for book in (getattr(model, 'books', {}) or {}).values():
            path = book.get('book_fpath') if isinstance(book, dict) else None
            if path:
                book_paths.append(path)
    except Exception:
        pass

    target_sheets = {rng.get('sheet', '').upper() for rng in ref.ranges}

    for path in book_paths:
        try:
            wb = openpyxl.load_workbook(path, data_only=False)
        except Exception:
            continue
        for ws, pivot in _iter_pivot_tables(wb):
            if ws.title.upper() not in target_sheets:
                continue
            location_ref = _pivot_location_ref(pivot)
            if not location_ref:
                continue
            if not _ranges_contains(ws.title, location_ref, ref):
                continue
            source = _pivot_source_ref(pivot)
            if source is None:
                continue
            sheet, src_ref = source
            try:
                return Ranges().push('{}!{}'.format(sheet, src_ref))
            except Exception:
                continue
    return None
