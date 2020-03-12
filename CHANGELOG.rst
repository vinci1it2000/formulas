Changelog
=========


v1.0.0 (2020-03-12)
-------------------

Feat
~~~~
- (core): Add `CODE_OF_CONDUCT.md`.

- (function) :gh:`39`: Transform `NotImplementedError` into `#NAME?`.

- (text) :gh:`39`: Add `CONCAT` and `CONCATENATE` functions.

- (logic) :gh:`38`: Add TRUE/FALSE functions.

- (excel) :gh:`42`: Save missing nodes.

- (excel) :gh:`42`: Update logic for `RangesAssembler`.

- (excel): Improve performance of `finish` method.

- (core): Update build script.

- (core): Add support for python 3.8 and drop python 3.5 and drop
  `appveyor`.

- (core): Improve memory performance.

- (refact): Update copyright.

- (operand): Add `fast_range2parts_v4` for named ranges.


Fix
~~~
- (math) :gh:`37`: Match excel default rounding algorithm of round half
  up.

- (cell): Correct reference in `push` method.

- (readme): Correct doctest.

- (token): Correct separator parser.

- (excel) :gh:`35`: Update logic to parse named ranges.

- (operand): Associate `excel_id==0` to current excel.

- (array): Ensure correct deepcopy of `Array` attributes.

- (operand) :gh:`39`: Correct range parser for named ranges.

- (operand) :gh:`41`: Correct named ranges parser.


v0.4.0 (2019-08-31)
-------------------

Feat
~~~~
- (doc): Add binder.

- (setup): Add env `ENABLE_SETUP_LONG_DESCRIPTION`.

- (core): Add useful constants.

- (excel): Add option to write all calculate books inside a folder.

- (stat) :gh:`21`: Add `COUNTBLANK`, `LARGE`, `SMALL` functions.

- (date) :gh:`35`: Add `NPV`, `XNPV`, `IRR`, `XIRR` functions.

- (stat) :gh:`21`: Add `AVERAGEIF`, `COUNT`, `COUNTA`, `COUNTIF`
  functions.

- (math) :gh:`21`: Add `SUMIF` function.

- (date) :gh:`21`, :gh:`35`, :gh:`36`: Add `date` functions `DATE`,
  `DATEVALUE`, `DAY`, `MONTH`, `YEAR`, `TODAY`, `TIME`, `TIMEVALUE`,
  `SECOND`, `MINUTE`, `HOUR`, `NOW`, `YEARFRAC`.

- (info) :gh:`21`: Add `NA` function.

- (date) :gh:`21`, :gh:`35`, :gh:`36`: Add `date` functions `DATE`,
  `DATEVALUE`, `DAY`, `MONTH`, `YEAR`, `TODAY`, `TIME`, `TIMEVALUE`,
  `SECOND`, `MINUTE`, `HOUR`, `NOW`, `YEARFRAC`.

- (stat) :gh:`35`: Add `MINA`, `AVERAGEA`, `MAXA` functions.


Fix
~~~
- (setup): Update tests requirements.

- (setup): Correct setup dependency (`beautifulsoup4`).

- (stat): Correct round indices.

- (setup) :gh:`34`: Build universal wheels.

- (test): Correct import error.

- (date) :gh:`35`: Correct behaviour of `LOOKUP` function when dealing
  with errors.

- (excel) :gh:`35`: Improve cycle detection.

- (excel,date) :gh:`21`, :gh:`35`: Add custom Excel Reader to parse raw
  datetime.

- (excel) :gh:`35`: Correct when definedName is relative `#REF!`.


v0.3.0 (2019-04-24)
-------------------

Feat
~~~~
- (logic) :gh:`27`: Add `OR`, `XOR`, `AND`, `NOT` functions.

- (look) :gh:`27`: Add `INDEX` function.

- (look) :gh:`24`: Improve performances of `look` functions.

- (functions) :gh:`26`: Add `SWITCH`.

- (functions) :gh:`30`: Add `GCD` and `LCM`.

- (chore): Improve performances avoiding `combine_dicts`.

- (chore): Improve performances checking intersection.


Fix
~~~
- (tokens): Correct string nodes ids format adding `"`.

- (ranges): Correct behaviour union of ranges.

- (import): Enable PyCharm autocomplete.

- (import): Save imports.

- (test): Add repo path to system path.

- (parser): Parse empty args for functions.

- (functions) :gh:`30`: Correct implementation of `GCD` and `LCM`.

- (ranges) :gh:`24`: Enable full column and row reference.

- (excel): Correct bugs due to new `openpyxl`.


v0.2.0 (2018-12-11)
-------------------

Feat
~~~~
- (doc) :gh:`23`: Enhance `ExcelModel` documentation.


Fix
~~~
- (core): Add python 3.7 and drop python 3.4.

- (excel): Make `ExcelModel` dillable and pickable.

- (builder): Avoid FormulaError exception during formulas compilation.

- (excel): Correct bug when compiling excel with circular references.


v0.1.4 (2018-10-19)
-------------------

Fix
~~~
- (tokens) :gh:`20`: Improve Number regex.


v0.1.3 (2018-10-09)
-------------------

Feat
~~~~
- (excel) :gh:`16`: Solve circular references.

- (setup): Add donate url.


Fix
~~~

- (functions) :gh:`18`: Enable `check_error` in `IF` function just for
  the first argument.

- (functions) :gh:`18`: Disable `input_parser` in `IF` function to
  return any type of values.

- (rtd): Define `fpath` from `prj_dir` for rtd.

- (rtd): Add missing requirements `openpyxl` for rtd.

- (setup): Patch to use `sphinxcontrib.restbuilder` in setup
  `long_description`.


Other
~~~~~
- Update documentation.

- Replace `excel` with `Excel`.

- Create PULL_REQUEST_TEMPLATE.md.

- Update issue templates.

- Update copyright.

- (doc): Update author mail.


v0.1.2 (2018-09-12)
-------------------

Feat
~~~~
- (functions) :gh:`14`: Add `ROW` and `COLUMN`.

- (cell): Pass cell reference when compiling cell + new function struct
  with dict to add inputs like CELL.

Fix
~~~
- (ranges): Replace system max size with excel max row and col.

- (tokens): Correct number regex.


v0.1.1 (2018-09-11)
-------------------

Feat
~~~~
- (contrib): Add contribution instructions.

- (setup): Add additional project_urls.

- (setup): Update `Development Status` to `4 - Beta`.


Fix
~~~

- (init) :gh:`15`: Replace `FUNCTIONS` and `OPERATORS` objs with
  `get_functions`, `SUBMODULES`.

- (doc): Correct link docs_status.


v0.1.0 (2018-07-20)
-------------------

Feat
~~~~
- (readme) :gh:`6`, :gh:`7`: Add examples.

- (doc): Add changelog.

- (test): Add info of executed test of `test_excel_model`.

- (functions) :gh:`11`: Add `HEX2OCT`, `HEX2BIN`, `HEX2DEC`, `OCT2HEX`,
  `OCT2BIN`, `OCT2DEC`, `BIN2HEX`, `BIN2OCT`, `BIN2DEC`, `DEC2HEX`,
  `DEC2OCT`, and `DEC2BIN` functions.

- (setup) :gh:`13`: Add extras_require to setup file.


Fix
~~~
- (excel): Use DispatchPipe to compile a sub model of excel workbook.

- (range) :gh:`11`: Correct range regex to avoid parsing of function
  like ranges (e.g., HEX2DEC).


v0.0.10 (2018-06-05)
--------------------

Feat
~~~~
- (look): Simplify `_get_type_id` function.


Fix
~~~
- (functions): Correct ImportError for FUNCTIONS.

- (operations): Correct behaviour of the basic operations.


v0.0.9 (2018-05-28)
-------------------

Feat
~~~~
- (excel): Improve performances pre-calculating the range format.

- (core): Improve performances using `DispatchPipe` instead
  `SubDispatchPipe` when compiling formulas.

- (function): Improve performances setting `errstate` outside
  vectorization.

- (core): Improve performances of range2parts function (overall 50%
  faster).


Fix
~~~
- (ranges): Minimize conversion str to int and vice versa.

- (functions) :gh:`10`: Avoid returning shapeless array.


v0.0.8 (2018-05-23)
-------------------

Feat
~~~~
- (functions): Add `MATCH`, `LOOKUP`, `HLOOKUP`, `VLOOKUP` functions.

- (excel): Add method to compile `ExcelModel`.

- (travis): Run coveralls in python 3.6.

- (functions): Add
  `FIND`,`LEFT`,`LEN`,`LOWER`,`MID`,`REPLACE`,`RIGHT`,`TRIM`, and`UPPER`
  functions.

- (functions): Add `IRR` function.

- (formulas): Custom reshape to Array class.

- (functions): Add `ISO.CEILING`, `SQRTPI`, `TRUNC` functions.

- (functions): Add `ROUND`, `ROUNDDOWN`, `ROUNDUP`, `SEC`, `SECH`,
  `SIGN` functions.

- (functions): Add `DECIMAL`, `EVEN`, `MROUND`, `ODD`, `RAND`,
  `RANDBETWEEN` functions.

- (functions): Add `FACT` and `FACTDOUBLE` functions.

- (functions): Add `ARABIC` and `ROMAN` functions.

- (functions): Parametrize function `wrap_ufunc`.

- (functions): Split function `raise_errors` adding `get_error`
  function.

- (ranges): Add custom default and error value for defining ranges
  Arrays.

- (functions): Add `LOG10` function + fix `LOG`.

- (functions): Add `CSC` and `CSCH` functions.

- (functions): Add `COT` and `COTH` functions.

- (functions): Add `FLOOR`, `FLOOR.MATH`, and `FLOOR.PRECISE` functions.

- (test): Improve log message of test cell.


Fix
~~~
- (rtd): Update installation file for read the docs.

- (functions): Remove unused functions.

- (formulas): Avoid too broad exception.

- (functions.math): Drop scipy dependency for calculate factorial2.

- (functions.logic): Correct error behaviour of `if` and `iferror`
  functions + add BroadcastError.

- (functions.info): Correct behaviour of `iserr` function.

- (functions): Correct error behaviour of average function.

- (functions): Correct `iserror` and `iserr` returning a custom Array.

- (functions): Now `xceiling` function returns np.nan instead
  Error.errors['#NUM!'].

- (functions): Correct `is_number` function, now returns False when
  number is a bool.

- (test): Ensure same order of workbook comparisons.

- (functions): Correct behaviour of `min` `max` and `int` function.

- (ranges): Ensure to have a value with correct shape.

- (parser): Change order of parsing to avoid TRUE and FALSE parsed as
  ranges or errors as strings.

- (function):Remove unused kwargs n_out.

- (parser): Parse error string as formulas.

- (readme): Remove `downloads_count` because it is no longer available.


Other
~~~~~
- Refact: Update Copyright + minor pep.

- Excel returns 1-indexed string positions???

- Added common string functions.

- Merge pull request :gh:`9` from ecatkins/irr.

- Implemented IRR function using numpy.


v0.0.7 (2017-07-20)
-------------------

Feat
~~~~
- (appveyor): Add python 3.6.

- (functions) :gh:`4`: Add `sumproduct` function.


Fix
~~~
- (install): Force update setuptools>=36.0.1.

- (functions): Correct `iserror` `iserr` functions.

- (ranges): Replace '#N/A' with '' as empty value when assemble values.

- (functions) :gh:`4`: Remove check in ufunc when inputs have different
  size.

- (functions) :gh:`4`: Correct `power`, `arctan2`, and `mod` error
  results.

- (functions) :gh:`4`: Simplify ufunc code.

- (test) :gh:`4`: Check that all results are in the output.

- (functions) :gh:`4`: Correct `atan2` argument order.

- (range) :gh:`5`: Avoid parsing function name as range when it is
  followed by `(`.

- (operator) :gh:`3`: Replace `strip` with `replace`.

- (operator) :gh:`3`: Correct valid operators like `^-` or `*+`.


Other
~~~~~
- Made the ufunc wrapper work with multi input functions, e.g., power,
  mod, and atan2.

- Created a workbook comparison method in TestExcelModel.

- Added MIN and MAX to the test.xlsx.

- Cleaned up the ufunc wrapper and added min and max to the functions
  list.

- Relaxed equality in TestExcelModel and made some small fixes to
  functions.py.

- Added a wrapper for numpy ufuncs, mapped some Excel functions to
  ufuncs and provided tests.


v0.0.6 (2017-05-31)
-------------------

Fix
~~~
- (plot): Update schedula to 0.1.12.

- (range): Sheet name without commas has this [^\W\d][\w\.] format.


v0.0.5 (2017-05-04)
-------------------

Fix
~~~
- (doc): Update schedula to 0.1.11.


v0.0.4 (2017-02-10)
-------------------

Fix
~~~
- (regex): Remove deprecation warnings.


v0.0.3 (2017-02-09)
-------------------

Fix
~~~
- (appveyor): Setup of lxml.

- (excel): Remove deprecation warning openpyxl.

- (requirements): Update schedula requirement 0.1.9.


v0.0.2 (2017-02-08)
-------------------

Fix
~~~
- (setup): setup fails due to long description.

- (excel): Remove deprecation warning `remove_sheet` --> `remove`.


