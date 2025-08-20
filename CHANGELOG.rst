Changelog
=========


v1.3.0 (2025-08-20)
-------------------

Feat
~~~~
- (setup): Add package_data.

- (excel): Improve readability of Excel comparison method.

- (functions): Add `ASC`, `BAHTTEXT`, `CLEAN`, `CHAR`, `UNICHAR`,
  `UNICODE`, `EXACT`, `FINDB`, `LEFTB`, `LENB`, `MIDB`, `NUMBERVALUE`,
  `PROPER`, `REGEXEXTRACT`, `REGEXREPLACE`, `REGEXTEST`, `REPLACEB`,
  `REPT`, `RIGHTB`, `SEARCHB`, `VALUETOTEXT`, `ARRAYTOTEXT`, `FIXED`,
  `TEXTSPLIT`, `TEXTAFTER`, `TEXTBEFORE`.

- (functions): Add `BESSELJ`, `BESSELI`, `BESSELK`, `BESSELY`, `BITAND`,
  `BITOR`, `BITXOR`, `BITLSHIFT`, `BITRSHIFT`, `CONVERT`, `ERF`,
  `ERF.PRECISE`, `ERFC`, `ERFC.PRECISE`, `DELTA`, `GESTEP`, `COMPLEX`,
  `IMDIV`, `IMSUB`, `IMSUM`, `IMPRODUCT`, `IMABS`, `IMREAL`,
  `IMAGINARY`, `IMARGUMENT`, `IMCONJUGATE`, `IMCOS`, `IMCOSH`, `IMCOT`,
  `IMCSC`, `IMCSCH`, `IMEXP`, `IMLN`, `IMLOG10`, `IMLOG2`, `IMSEC`,
  `IMSECH`, `IMSIN`, `IMSINH`, `IMSQRT`, `IMTAN`, `IMPOWER`.

- (operators): Add support for Implicit intersection operator `@`.

- (core): Add `.ods` file support.


Fix
~~~
- (excel): Correct external link of `.ods` files.

- (setup): Add missing requirements.

- (excel): Parse properly dynamic arrays also from JSON format.

- (look): Correct `FILTER` function was changing input values.

- (text): Correct `T` function behaviour.

- (token): Correct compile error for excel # `Errors`.

- (operators): Correct behaviour of `=` operator `"A"="a"` return now
  `TRUE` like excel.

- (excel): Correct handling of Excel Illegal Character.


Other
~~~~~
- Update test cases to improve coverage.

- Add new test cases for `.ods` file.

- Add new test cases for all new functions.


v1.2.11 (2025-07-28)
--------------------

Fix
~~~
- (test): Update test cases.

- (look) :gh:`170`: Correct  `MATCH` and `SUMIFS` behaviour.

- (math): Update `RANDBETWEEN` error handling.

- (look): Correct INDEX function bug beacuse of new numpy 2.4.

- (doc): Correct readme badges.


v1.2.10 (2025-05-21)
--------------------

Feat
~~~~
- (functions): Add `EOMONTH`, `SUMIFS`, `AVERAGEIFS`, `COUNTIFS`,
  `MAXIFS`, `MINIFS`.


v1.2.9 (2025-04-05)
-------------------

Feat
~~~~
- (core): Update test cases.

- (core): Update python versions.

- (excel): Add `InvRangesAssembler`.

- (functions): Add `FILTER`, `TRANSPOSE`, `SUBSTITUTE`, `TEXTJOIN`, `T`.

- (doc): Add JetBrains sponsor.

- (look): Improve performances of `MATCH` and `LOOKUP`.

- (math): Add `MDETERM`, `MINVERSE`, and `MMULT` functions.

- (core): Add ANCHORARRAY functionality.


Fix
~~~
- (text) :gh:`146`: Correct TEXT function date formatter logic.

- (functions) :gh:`147`: Correct array collapse behaviour.

- (text) :gh:`149`, :gh:`158`: Add default value of LEFT and RIGHT
  functions.

- (core) :gh:`159`: Correct parsing error.

- (doc): Correct doctests.

- (text): Add missing function `_XLFN.CONCATENATE`.

- (doc): Correct documentation issue.


v1.2.8 (2024-07-16)
-------------------

Feat
~~~~
- (core): Update Copyright.

- (functions) :gh:`109`, :gh:`111`, :gh:`124`, :gh:`125`: Update test
  cases.

- (stat) :gh:`111`: Add `PERCENTILE`, `PERCENTILE.INC`, and
  `PERCENTILE.EXC` functions.

- (stat) :gh:`111`: Add `NORM.S.DIST`, `NORM.S.INV`, `NORM.DIST`,
  `NORM.INV`,`NORMDIST`, `NORMINV`,`NORMSINV` functions.

- (stat) :gh:`111`: Add `NORMSDIST` function.

- (stat) :gh:`124`: Correct implementation `QUARTILE` and add
  `QUARTILE.INC` and `QUARTILE.EXC`.

- (functions) :gh:`124`: Add `QUARTILE` to stat functions.

- (functions) :gh:`125`: Add `SUMSQ` to stat functions.

- (tokens) :gh:`139`: Allow last parameters to be empty in a function
  call.

- (tokens) :gh:`139`: Allow first param to be empty.

- (core): Update `.gitignore` settings.

- (text): Add `CODE` function.

- (text): Add `CHAR` function.

- (test): Update coverage python version.


Fix
~~~
- (test) :gh:`111`: Correct test case for windows.

- (excel) :gh:`109`: Correct parser for named range with backslash in
  name.

- (functions) :gh:`125`: Move `SUMSQ` function to math.

- (core): Correct repr formatting of ranges for numpy version 2.x.

- (tokens) :gh:`145`: Correct handling of `#REF!` when compiling
  functions.

- (text): Correct `CODE` function.

- (text): Add `CODE` text case.

- (excel) :gh:`132`: Correction on how to handle the empty values used
  within a formula.

- (excel): Add `#EMPTY` value to save correctly the model as dict.

- (excel) :gh:`134`, :gh:`135`: Correct `inverse_references` handling
  when model defined with `from_dict`.

- (excel): Correct tolerance.

- (setup): Correct setup config file.


v1.2.7 (2023-11-14)
-------------------

Feat
~~~~
- (builder) :gh:`104`: Allow custom reference definition.

- (test): Update test cases.

- (operand) :gh:`106`: Accept number like `.3` to be parsed.

- (text) :gh:`113`: Add `TEXT` function without fraction formatting.

- (logic): Update logic functions according to new excel logic.

- (text) :gh:`113`: Add `VALUE` function.

- (math) :gh:`121`: Improve performances of `SUMPRODUCT`, `PRODUCT`,
  `SUM`, and `SUMIF`.

- (setup): Update requirements.

- (core): Change development status.

- (core): Add support for python 3.10 and 3.11.

- (functions) :gh:`121`: Improve handling of EMPTY values.

- (excel): Avoid using `flatten` function in basic routines.

- (doc): Add Read the Docs configuration file.

- (excel): Add tolerance when comparing two excels.

- (excel): Add compare method to verify if formulas is able to replicate
  excel values.


Fix
~~~
- (doc): Remove broken badge.

- (excel) :gh:`100`: Correct reading rounding from excel.

- (math) :gh:`100`: Correct `TRUNC` defaults.

- (tokens) :gh:`113`: Correct `sheet_id` definition.

- (functions): Correct dill pickling error.

- (excel): Correct reference parsing when loading from JSON.

- (functions): Use an alternative method of vectorize when more than 32
  arguments are provided.

- (look): Correct `MATCH`, `LOOKUP`,`HLOOKUP`, and `VLOOKUP` behaviour
  when empty values are given.

- (date): Correct `DATEDIF` behaviour when unit is lowercase.

- (test): Use regex for unstable tests due to changes in last digits.

- (doc): Correct documentation bug due to new `sphinx`.

- (excel) :gh:`114`: Update reading code according to `openpyxl>=3.1`.


v1.2.6 (2022-12-13)
-------------------

Fix
~~~
- (setup): Update `schedula` requirement.


v1.2.5 (2022-11-07)
-------------------

Fix
~~~
- (parser): Correct missing raise.

- (excel): Skip hidden named ranges.


v1.2.4 (2022-07-02)
-------------------

Feat
~~~~
- (core): Improve speed performance.

- (cell): Improve speed `RangesAssembler` definition.


Fix
~~~
- (cell): Correct range assembler defaults when no `sheet_id` is
  defined.

- (math) :gh:`99`: Convert args into np.arrays in func `xsumproduct`.

- (look): Correct lookup parser for float and strings.


v1.2.3 (2022-05-10)
-------------------

Feat
~~~~
- (test): Add more error logs.

- (test): Improve code coverage.

- (builder): Add `compile_class` attribute to `AstBuilder`.

- (info): Add `ISODD`, `ISEVEN`, `ISBLANK`, `ISTEXT`, `ISNONTEXT`, and
  `ISLOGICAL` functions.


Fix
~~~
- (excel): Correct file path excel definition.

- (logic): Correct `SWITCH` error handling.

- (actions): Rename workflow name.

- (readme): Correct badge link for dependencies status.

- (excel): Correct `basedir` reference to load files.

- (date): Correct `YEARFRAC` and `DATEDIF` formulation.

- (cell): Enable R1C1 notation for absolute and relative references.

- (cell): Correct RangeAssembler value assignment.


v1.2.2 (2022-01-22)
-------------------

Fix
~~~
- (excel): Correct function compilation from excel.


v1.2.1 (2022-01-21)
-------------------

Feat
~~~~
- (functions): Improve performances caching results.

- (excel): Make replacing missing ref optional in `from_dict` method.

- (excel) :gh:`73`, :gh:`75`: Improve performances to parse full ranges.


Fix
~~~
- (excel): Correct compile function when inputs are computed with a
  default function.


v1.2.0 (2021-12-23)
-------------------

Feat
~~~~
- (binder): Refresh environment binder for 2021.

- (look) :gh:`87`: Add `ADDRESS` function.

- (test): Update test cases.

- (financial) :gh:`74`, :gh:`87`: Add `FV`, `PV`, `IPMT`, `PMT`, `PPMT`,
  `RATE`, `CUMIPMT`, and `NPER` functions.

- (info, logic): Add `ISNA` and `IFNA` functions.

- (date) :gh:`87`: Add `WEEKDAY`, `WEEKNUM`, `ISOWEEKNUM`, and `DATEDIF`
  functions.

- (stat, math) :gh:`87`: Add `SLOPE` and `PRODUCT` functions.

- (stats) :gh:`87`: Add `CORREL` and `MEDIAN` functions.

- (bin): Add `bin` folder.

- (actions): Add test cases.

- (stats) :gh:`80`: Add `FORECAST` and `FORECAST.LINEAR` functions.

- (excel) :gh:`82`: Add inverse of simple references.


Fix
~~~
- (stat): Correct `LARGE` and `SMALL` error handling.

- (actions): Skip `Setup Graphviz` when not needed.

- (actions): Correct coverall setting.

- (actions): Remove unstable test case.

- (actions): Disable fail fast.

- (date, stat): Correct collapsed return value.

- (function) :gh:`78`, :gh:`79`, :gh:`91`: Correct import error.


v1.1.1 (2021-10-13)
-------------------

Feat
~~~~
- (excel): Improve performances of `complete` method.

- (setup): Add add python 3.9 in setup.py.

- (functions): Add `SEARCH`, `ISNUMBER`, and `EDATE` functions.

- (travis): Update python version for coveralls.


Fix
~~~
- (doc): Correct missing documentation link.

- (doc): Correct typo.

- (operator) :gh:`70`: Correct `%` operator preceded by space.


v1.1.0 (2021-02-16)
-------------------

Feat
~~~~
- (look) :gh:`57`: Add `SINGLE` function.

- (function) :gh:`51`: Add google Excel functions.

- (logic) :gh:`55`, :gh:`57`: Add IFS function.

- (excel) :gh:`65`: Add documentation and rename method to load models
  from ranges.

- (excel) :gh:`65`: Add method to load sub-models from range.

- (doc): Update Copyright.

- (excel): Improve performances.

- (excel) :gh:`64`: Read model from outputs.

- (core): Update range definition with path file.

- (excel) :gh:`64`: Add warning for missing reference.

- (excel) :gh:`64`: Add warning message when book loading fails.

- (readme) :gh:`44`: Add example to export and import the model to JSON
  format.

- (readme) :gh:`53`: Add instructions to install the development
  version.

- (excel) :gh:`44`: Add feature to export and import the model to JSON-
  able dict.

- (stat, comp) :gh:`43`: Add `STDEV`, `STDEV.S`, `STDEV.P`, `STDEVA`,
  `STDEVPA`, `VAR`, `VAR.S`, `VAR.P`, `VARA`, and `VARPA` functions.


Fix
~~~
- (financial): Correct requirements for `irr` function.

- (excel) :gh:`48`: Correct reference pointing to different workbooks.

- (function) :gh:`67`: Correct compilation of impure functions (e.g.,
  `rand`, `now`, etc.).

- (look) :gh:`66`: Correct `check` function did not return value.

- (test): Remove `temp` dir.

- (excel): Correct external link reading.

- (operator) :gh:`63`: Correct operator parser when starts with spaces.

- (text) :gh:`61`: Convert float as int when stringify if it is an
  integer.

- (math) :gh:`59`: Convert string to number in math operations.

- (functions): Correct `_xfilter` operating range type.

- (parser) :gh:`61`: Skip `\n` in formula expression.

- (operator) :gh:`58`: Correct operator parser for composed operators.

- (excel): Correct invalid range definition and missing sheet or files.

- (operand) :gh:`52`: Correct range parser.

- (operand) :gh:`50`: Correct sheet name parser with space.

- (tokens): Correct closure parenthesis parser.

- (excel): Skip function compilation for string cells.

- (tokens): Correct error parsing when sheet name is defined.


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


