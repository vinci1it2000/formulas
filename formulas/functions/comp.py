#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2016-2025 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
Python equivalents of compatibility Excel functions.
"""

from . import Error, FoundError
from .stat import FUNCTIONS as FSTAT
from .stat import (
    _convert_args,
    wrap_ufunc,
    xbetadist,
    xhypergeom_dist,
    xlognormdist,
    xnegbinomdist,
    xt_dist2t,
    xt_distrt,
)

FUNCTIONS = {
    "BETAINV": FSTAT["BETA.INV"],
    "BINOMDIST": FSTAT["BINOM.DIST"],
    "CHIDIST": FSTAT["CHISQ.DIST.RT"],
    "CHIINV": FSTAT["CHISQ.INV.RT"],
    "CHITEST": FSTAT["CHISQ.TEST"],
    "CONFIDENCE": FSTAT["CONFIDENCE.NORM"],
    "COVAR": FSTAT["COVARIANCE.P"],
    "CRITBINOM": FSTAT["BINOM.INV"],
    "EXPONDIST": FSTAT["EXPON.DIST"],
    "FDIST": FSTAT["F.DIST.RT"],
    "FINV": FSTAT["F.INV.RT"],
    "FTEST": FSTAT["F.TEST"],
    "GAMMADIST": FSTAT["GAMMA.DIST"],
    "GAMMAINV": FSTAT["GAMMA.INV"],
    "LOGINV": FSTAT["LOGNORM.INV"],
    "MODE": FSTAT["MODE.SNGL"],
    "NORMDIST": FSTAT["NORM.DIST"],
    "NORMINV": FSTAT["NORM.INV"],
    "NORMSDIST": FSTAT["NORM.S.DIST"],
    "NORMSINV": FSTAT["NORM.S.INV"],
    "PERCENTILE": FSTAT["PERCENTILE.INC"],
    "PERCENTRANK": FSTAT["PERCENTRANK.INC"],
    "POISSON": FSTAT["POISSON.DIST"],
    "QUARTILE": FSTAT["QUARTILE.INC"],
    "RANK": FSTAT["RANK.EQ"],
    "STDEV": FSTAT["STDEV.S"],
    "STDEVP": FSTAT["STDEV.P"],
    "TINV": FSTAT["T.INV.2T"],
    "TTEST": FSTAT["T.TEST"],
    "VAR": FSTAT["VAR.S"],
    "VARP": FSTAT["VAR.P"],
    "WEIBULL": FSTAT["WEIBULL.DIST"],
    "ZTEST": FSTAT["Z.TEST"],
}
FUNCTIONS["BETADIST"] = wrap_ufunc(
    xbetadist,
    input_parser=lambda *a: tuple(map(_convert_args, a[:3]))
    + (1,)
    + tuple(map(_convert_args, a[3:])),
)
FUNCTIONS["HYPGEOMDIST"] = wrap_ufunc(
    xhypergeom_dist, input_parser=lambda *a: tuple(map(_convert_args, a)) + (0,)
)
FUNCTIONS["LOGNORMDIST"] = wrap_ufunc(
    xlognormdist, input_parser=lambda *a: tuple(map(_convert_args, a))
)
FUNCTIONS["NEGBINOMDIST"] = wrap_ufunc(
    xnegbinomdist, input_parser=lambda *a: tuple(map(_convert_args, a)) + (0,)
)


def xtdist(x, deg_freedom, tails):
    tails = int(tails)
    if tails == 1:
        return xt_distrt(x, deg_freedom)
    elif tails == 2:
        return xt_dist2t(x, deg_freedom)
    raise FoundError(err=Error.errors["#NUM!"])


FUNCTIONS["TDIST"] = wrap_ufunc(
    xtdist, input_parser=lambda *a: tuple(map(_convert_args, a))
)
