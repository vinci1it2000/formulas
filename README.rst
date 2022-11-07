.. _start-quick:

##################################################
formulas: An Excel formulas interpreter in Python.
##################################################
|pypi_ver| |test_status| |cover_status| |docs_status| |dependencies|
|github_issues| |python_ver| |proj_license| |binder|

:release:       1.2.5
:date:          2022-11-07 13:45:00
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

Development version
-------------------
To help with the testing and the development of `formulas`, you can install the
development version:

.. code-block:: console

    $ pip install https://github.com/vinci1it2000/formulas/archive/dev.zip

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
    Array(7.0, dtype=object)
    >>> func.plot(workflow=True, view=False)  # Set view=True to plot in the default browser.
    SiteMap([(=((1 + 1) + (B3 / A2)), SiteMap())])

Excel workbook
--------------
An example how to load, calculate, and write an Excel workbook is the following:

.. testsetup::

    >>> import os.path as osp
    >>> from setup import mydir
    >>> fpath = osp.join(mydir, 'test/test_files/excel.xlsx')
    >>> dir_output = osp.join(mydir, 'test/test_files/tmp')

.. doctest::

    >>> import formulas
    >>> fpath, dir_output = 'excel.xlsx', 'output'  # doctest: +SKIP
    >>> xl_model = formulas.ExcelModel().loads(fpath).finish()
    >>> xl_model.calculate()
    Solution(...)
    >>> xl_model.write(dirpath=dir_output)
    {'EXCEL.XLSX': {Book: <openpyxl.workbook.workbook.Workbook ...>}}

.. tip:: If you have or could have **circular references**, add `circular=True`
   to `finish` method.

To plot the dependency graph that depict relationships between Excel cells:

.. dispatcher:: dsp
   :code:

    >>> dsp = xl_model.dsp
    >>> dsp.plot(view=False)  # Set view=True to plot in the default browser.
    SiteMap([(ExcelModel, SiteMap(...))])

To overwrite the default inputs that are defined by the excel file or to impose
some value to a specific cell:

    >>> xl_model.calculate(
    ...     inputs={
    ...         "'[excel.xlsx]'!INPUT_A": 3,  # To overwrite the default value.
    ...         "'[excel.xlsx]DATA'!B3": 1  # To impose a value to B3 cell.
    ...     },
    ...     outputs=[
    ...        "'[excel.xlsx]DATA'!C2", "'[excel.xlsx]DATA'!C4"
    ...     ] # To define the outputs that you want to calculate.
    ... )
    Solution([("'[excel.xlsx]'!INPUT_A", <Ranges>('[excel.xlsx]DATA'!A2)=[[3]]),
              ("'[excel.xlsx]DATA'!B3", <Ranges>('[excel.xlsx]DATA'!B3)=[[1]]),
              ("'[excel.xlsx]DATA'!A2", <Ranges>('[excel.xlsx]DATA'!A2)=[[3]]),
              ("'[excel.xlsx]DATA'!A3", <Ranges>('[excel.xlsx]DATA'!A3)=[[6]]),
              ("'[excel.xlsx]DATA'!D2", <Ranges>('[excel.xlsx]DATA'!D2)=[[1]]),
              ("'[excel.xlsx]'!INPUT_B", <Ranges>('[excel.xlsx]DATA'!A3)=[[6]]),
              ("'[excel.xlsx]DATA'!B2", <Ranges>('[excel.xlsx]DATA'!B2)=[[9.0]]),
              ("'[excel.xlsx]DATA'!D3", <Ranges>('[excel.xlsx]DATA'!D3)=[[2.0]]),
              ("'[excel.xlsx]DATA'!C2", <Ranges>('[excel.xlsx]DATA'!C2)=[[10.0]]),
              ("'[excel.xlsx]DATA'!D4", <Ranges>('[excel.xlsx]DATA'!D4)=[[3.0]]),
              ("'[excel.xlsx]DATA'!C4", <Ranges>('[excel.xlsx]DATA'!C4)=[[4.0]])])


To build a single function out of an excel model with fixed inputs and outputs,
you can use the `compile` method of the `ExcelModel` that returns a
DispatchPipe_. This is a function where the inputs and outputs are defined by
the data node ids (i.e., cell references).

.. dispatcher:: func
   :code:

    >>> func = xl_model.compile(
    ...     inputs=[
    ...         "'[excel.xlsx]'!INPUT_A",  # First argument of the function.
    ...         "'[excel.xlsx]DATA'!B3"   # Second argument of the function.
    ...     ], # To define function inputs.
    ...     outputs=[
    ...         "'[excel.xlsx]DATA'!C2", "'[excel.xlsx]DATA'!C4"
    ...     ] # To define function outputs.
    ... )
    >>> func
    <schedula.utils.dsp.DispatchPipe object at ...>
    >>> [v.value[0, 0] for v in func(3, 1)]  # To retrieve the data.
    [10.0, 4.0]
    >>> func.plot(view=False)  # Set view=True to plot in the default browser.
    SiteMap([(ExcelModel, SiteMap(...))])

.. _DispatchPipe: https://schedula.readthedocs.io/en/master/_build/schedula/utils/dsp/schedula.utils.dsp.DispatchPipe.html#schedula.utils.dsp.DispatchPipe

Alternatively, to load a partial excel model from the output cells, you can use
the `from_ranges` method of the `ExcelModel`:

.. dispatcher:: dsp
   :code:

    >>> xl = formulas.ExcelModel().from_ranges(
    ...     "'[%s]DATA'!C2:D2" % fpath,  # Output range.
    ...     "'[%s]DATA'!B4" % fpath,  # Output cell.
    ... )
    >>> dsp = xl.dsp
    >>> sorted(dsp.data_nodes)
    ["'[excel.xlsx]'!INPUT_A",
     "'[excel.xlsx]'!INPUT_B",
     "'[excel.xlsx]'!INPUT_C",
     "'[excel.xlsx]DATA'!A2",
     "'[excel.xlsx]DATA'!A3",
     "'[excel.xlsx]DATA'!A3:A4",
     "'[excel.xlsx]DATA'!A4",
     "'[excel.xlsx]DATA'!B2",
     "'[excel.xlsx]DATA'!B3",
     "'[excel.xlsx]DATA'!B4",
     "'[excel.xlsx]DATA'!C2",
     "'[excel.xlsx]DATA'!D2"]


JSON export/import
~~~~~~~~~~~~~~~~~~
The `ExcelModel` can be exported/imported to/from a readable JSON format. The
reason of this functionality is to have format that can be easily maintained
(e.g. using version control programs like `git`). Follows an example on how to
export/import to/from JSON an `ExcelModel`:

.. testsetup::

    >>> import formulas
    >>> import os.path as osp
    >>> from setup import mydir
    >>> fpath = osp.join(mydir, 'test/test_files/excel.xlsx')
    >>> xl_model = formulas.ExcelModel().loads(fpath).finish()

.. doctest::

    >>> import json
    >>> xl_dict = xl_model.to_dict()  # To JSON-able dict.
    >>> xl_dict  # Exported format. # doctest: +SKIP
    {
     "'[excel.xlsx]DATA'!A1": "inputs",
     "'[excel.xlsx]DATA'!B1": "Intermediate",
     "'[excel.xlsx]DATA'!C1": "outputs",
     "'[excel.xlsx]DATA'!D1": "defaults",
     "'[excel.xlsx]DATA'!A2": 2,
     "'[excel.xlsx]DATA'!D2": 1,
     "'[excel.xlsx]DATA'!A3": 6,
     "'[excel.xlsx]DATA'!A4": 5,
     "'[excel.xlsx]DATA'!B2": "=('[excel.xlsx]DATA'!A2 + '[excel.xlsx]DATA'!A3)",
     "'[excel.xlsx]DATA'!C2": "=(('[excel.xlsx]DATA'!B2 / '[excel.xlsx]DATA'!B3) + '[excel.xlsx]DATA'!D2)",
     "'[excel.xlsx]DATA'!B3": "=('[excel.xlsx]DATA'!B2 - '[excel.xlsx]DATA'!A3)",
     "'[excel.xlsx]DATA'!C3": "=(('[excel.xlsx]DATA'!C2 * '[excel.xlsx]DATA'!A2) + '[excel.xlsx]DATA'!D3)",
     "'[excel.xlsx]DATA'!D3": "=(1 + '[excel.xlsx]DATA'!D2)",
     "'[excel.xlsx]DATA'!B4": "=MAX('[excel.xlsx]DATA'!A3:A4, '[excel.xlsx]DATA'!B2)",
     "'[excel.xlsx]DATA'!C4": "=(('[excel.xlsx]DATA'!B3 ^ '[excel.xlsx]DATA'!C2) + '[excel.xlsx]DATA'!D4)",
     "'[excel.xlsx]DATA'!D4": "=(1 + '[excel.xlsx]DATA'!D3)"
    }
    >>> xl_json = json.dumps(xl_dict, indent=True)  # To JSON.
    >>> xl_model = formulas.ExcelModel().from_dict(json.loads(xl_json))  # From JSON.

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
.. |test_status| image:: https://github.com/vinci1it2000/formulas/actions/workflows/tests.yml/badge.svg?branch=master
    :alt: Build status
    :target: https://github.com/vinci1it2000/formulas/actions/workflows/tests.yml?query=branch%3Amaster

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

.. |dependencies| image:: https://requires.io/github/vinci1it2000/formulas/requirements.svg?branch=master
    :target: https://requires.io/github/vinci1it2000/formulas/requirements/?branch=master
    :alt: Dependencies up-to-date?

.. |binder| image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/vinci1it2000/formulas/master?urlpath=lab%2Ftree%2Fbinder%2Findex.ipynb
    :alt: Live Demo
.. _end-badges:
