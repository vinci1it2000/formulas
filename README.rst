.. _start-quick:

##################################################
formulas: An Excel formulas interpreter in Python.
##################################################
|pypi_ver| |travis_status| |appveyor_status| |cover_status| |docs_status|
|dependencies| |github_issues| |python_ver| |proj_license|

:release:       0.1.3
:date:          2018-10-09 14:30:00
:repository:    https://github.com/vinci1it2000/formulas
:pypi-repo:     https://pypi.org/project/formulas/
:docs:          http://formulas.readthedocs.io/
:wiki:          https://github.com/vinci1it2000/formulas/wiki/
:download:      http://github.com/vinci1it2000/formulas/releases/
:donate:        https://donorbox.org/formulas
:keywords:      excel, formulas, interpreter, compiler, dispatch
:developers:    .. include:: AUTHORS.rst
:license:       `EUPL 1.1+ <https://joinup.ec.europa.eu/software/page/eupl>`_

.. _start-pypi:
.. _start-intro:

What is formulas?
=================
**formulas** implements an interpreter for Excel formulas, which parses and
compile Excel formulas expressions.

Moreover, it compiles Excel workbooks to python and executes without using the
Excel COM server. Hence, **Excel is not needed**.


Installation
============
To install it use (with root privileges):

.. code-block:: console

    $ pip install formulas

Or download the last git version and use (with root privileges):

.. code-block:: console

    $ python setup.py install


Install extras
--------------
Some additional functionality is enabled installing the following extras:

- excel: enables to compile Excel workbooks to python and execute using:
  :class:`~formulas.excel.ExcelModel`.
- plot: enables to plot the formula ast and the Excel model.

To install formulas and all extras, do:

.. code-block:: console

    $ pip install formulas[all]

.. _end-quick:

Basic Examples
==============
The following sections will show how to:

- parse a Excel formulas;
- load, compile, and execute a Excel workbook;
- extract a sub-model from a Excel workbook;
- add a custom function.

Parsing formula
---------------
An example how to parse and execute an Excel formula is the following:

    >>> import formulas
    >>> func = formulas.Parser().ast('=(1 + 1) + B3 / A2')[1].compile()

To visualize formula model and get the input order you can do the following:

.. dispatcher:: func
   :opt: graph_attr={'ratio': '1'}
   :code:

    >>> list(func.inputs)
    ['A2', 'B3']
    >>> func.plot(view=False)  # Set view=True to plot in the default browser.
    SiteMap([(=((1 + 1) + (B3 / A2)), SiteMap())])

Finally to execute the formula and plot the workflow:

.. dispatcher:: func
   :opt: workflow=True, graph_attr={'ratio': '1'}
   :code:

    >>> func(1, 5)
    OperatorArray(7.0, dtype=object)
    >>> func.plot(workflow=True, view=False)  # Set view=True to plot in the default browser.
    SiteMap([(=((1 + 1) + (B3 / A2)), SiteMap())])

Excel workbook
--------------
An example how to load, calculate, and write an Excel workbook is the following:

.. testsetup::

    >>> import os.path as osp
    >>> from setup import mydir
    >>> fpath = osp.join(mydir, 'test/test_files/excel.xlsx')

.. doctest::

    >>> import formulas
    >>> fpath = 'file.xlsx'  # doctest: +SKIP
    >>> xl_model = formulas.ExcelModel().loads(fpath).finish()
    >>> xl_model.calculate()
    Solution(...)
    >>> xl_model.write()
    {'EXCEL.XLSX': {Book: <openpyxl.workbook.workbook.Workbook ...>}}

.. tip:: If you have or could have **circular references**, add `circular=True`
   to `finish` method.

To plot the dependency graph that depict relationships between Excel cells:

.. dispatcher:: dsp
   :code:

    >>> dsp = xl_model.dsp
    >>> dsp.plot(view=False)  # Set view=True to plot in the default browser.
    SiteMap([(Dispatcher ..., SiteMap())])

To compile, execute, and plot a Excel sub-model you can do the following:

.. dispatcher:: func
   :code:

    >>> inputs = ["'[EXCEL.XLSX]DATA'!A2"]  # input cells
    >>> outputs = ["'[EXCEL.XLSX]DATA'!C2"]  # output cells
    >>> func = xl_model.compile(inputs, outputs)
    >>> func(2).value[0,0]
    4.0
    >>> func.plot(view=False)  # Set view=True to plot in the default browser.
    SiteMap([(Dispatcher ..., SiteMap())])

Custom functions
----------------
An example how to add a custom function to the formula parser is the following:

    >>> import formulas
    >>> FUNCTIONS = formulas.get_functions()
    >>> FUNCTIONS['MYFUNC'] = lambda x, y: 1 + y + x
    >>> func = formulas.Parser().ast('=MYFUNC(1, 2)')[1].compile()
    >>> func()
    4

.. _end-pypi:

Next moves
==========
Things yet to do: implement the missing Excel formulas.

.. _end-intro:
.. _start-badges:
.. |travis_status| image:: https://travis-ci.org/vinci1it2000/formulas.svg?branch=master
    :alt: Travis build status
    :target: https://travis-ci.org/vinci1it2000/formulas

.. |appveyor_status| image:: https://ci.appveyor.com/api/projects/status/i3bmqdc92u1bskg5?svg=true
    :alt: Appveyor build status
    :target: https://ci.appveyor.com/project/vinci1it2000/formulas

.. |cover_status| image:: https://coveralls.io/repos/github/vinci1it2000/formulas/badge.svg?branch=master
    :target: https://coveralls.io/github/vinci1it2000/formulas?branch=master
    :alt: Code coverage

.. |docs_status| image:: https://readthedocs.org/projects/formulas/badge/?version=stable
    :alt: Documentation status
    :target: https://formulas.readthedocs.io/en/stable/?badge=stable

.. |pypi_ver| image::  https://img.shields.io/pypi/v/formulas.svg?
    :target: https://pypi.python.org/pypi/formulas/
    :alt: Latest Version in PyPI

.. |python_ver| image:: https://img.shields.io/pypi/pyversions/formulas.svg?
    :target: https://pypi.python.org/pypi/formulas/
    :alt: Supported Python versions

.. |github_issues| image:: https://img.shields.io/github/issues/vinci1it2000/formulas.svg?
    :target: https://github.com/vinci1it2000/formulas/issues
    :alt: Issues count

.. |proj_license| image:: https://img.shields.io/badge/license-EUPL%201.1%2B-blue.svg?
    :target: https://raw.githubusercontent.com/vinci1it2000/formulas/master/LICENSE.txt
    :alt: Project License

.. |dependencies| image:: https://img.shields.io/requires/github/vinci1it2000/formulas.svg?
    :target: https://requires.io/github/vinci1it2000/formulas/requirements/?branch=master
    :alt: Dependencies up-to-date?
.. _end-badges:
