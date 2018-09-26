Contributing to formulas
========================

If you want to contribute to **formulas** and make it better, your help is very
welcome. The contribution should be sent by a *pull request*. Next sections will
explain how to implement and submit a new excel function:

- clone the repository
- implement a new function/functionality
- open a pull request

Clone the repository
--------------------
The first step to contribute to **formulas** is to clone the repository:

- Create a personal `fork <https://help.github.com/articles/fork-a-repo/
  #fork-an-example-repository>`_ of the `formulas <https://github.com/
  vinci1it2000/formulas>`_ repository on Github.
- `Clone <https://help.github.com/articles/fork-a-repo/
  #step-2-create-a-local-clone-of-your-fork>`_ the fork on your local machine.
  Your remote repo on Github is called ``origin``.
- `Add <https://help.github.com/articles/fork-a-repo/#step-3-configure-git-to
  -sync-your-fork-with-the-original-spoon-knife-repository>`_
  the original repository as a remote called ``upstream``, to maintain updated
  your fork.
- If you created your fork a while ago be sure to pull ``upstream`` changes into
  your local repository.
- Create a new branch to work on! Branch from ``dev``.

How to implement a new function
-------------------------------
Before coding, `study <https://support.office.com/en-us/article/
excel-functions-alphabetical-b3944572-255d-4efb-bb96-c6d90033e188>`_
the Excel function that you want to implement. If there is something similar
implemented in **formulas**, try to get inspired by the implemented code (I mean,
not reinvent the wheel) and to use ``numpy``. Follow the code style of the
project, including indentation. Add or change the documentation as needed.
Make sure that you have implemented the **full function syntax**, including the
`array syntax <https://support.office.com/en-us/article/guidelines-and
-examples-of-array-formulas-7d94a64e-3ff3-4686-9372-ecfd5caa57c7>`_.

Test cases are very important. This library uses a data-driven testing approach.
To implement a new function I recommend the `test-driven development cycle
<https://en.wikipedia.org/wiki/Test-driven_development
#Test-driven_development_cycle>`_. Hence, when you implement a new function,
you should write new test cases in ``test_cell/TestCell.test_output`` suite to
execute in the *cycle loop*. When you think that the code is ready, add new raw
test in ``test/test_files/test.xlsx`` (please follow the standard used for other
functions) and run the ``test_excel/TestExcelModel.test_excel_model``. This
requires more time but is needed to test the **array syntax** and to check if
the Excel documentation respects the reality.

When all test cases are ok (``python setup.py test``), open a pull request.

Do do list:

- Study the excel function syntax and behaviour when used as array formula.
- Check if there is something similar implemented in formulas.
- Implement/fix your feature, comment your code.
- Write/adapt tests and run them!

.. tip:: Excel functions are categorized by their functionality. If you are
  implementing a new functionality group, add a new module in
  ``formula/function`` and in ``formula.function.SUBMODULES`` and a new
  worksheet in ``test/test_files/test.xlsx`` (please respect the format).

.. note:: A pull request without new test case will not be taken into
   consideration.

How to open a pull request
--------------------------
Well done! Your contribution is ready to be submitted:

- Squash your commits into a single commit with git's
  `interactive rebase <https://help.github.com/articles/interactive-rebase>`_.
  Create a new branch if necessary. Always write your commit messages in the
  present tense. Your commit message should describe what the commit, when
  applied, does to the code – not what you did to the code.
- `Push <https://help.github.com/articles/pushing-to-a-remote/>`_ your branch to
  your fork on Github (i.e., ``git push origin dev``).
- From your fork `open <https://help.github.com/articles/creating-a-pull-
  request-from-a-fork/>`_ a *pull request* in the correct branch.
  Target the project's ``dev`` branch!
- Once the *pull request* is approved and merged you can pull the changes from
  ``upstream`` to your local repo and delete your extra branch(es).


Donate
======

If you want to `support <https://donorbox.org/formulas>`_ the **formulas**
development please donate and add your excel function preferences. The selection
of the functions to be implemented is done considering the cumulative donation
amount per function collected by the campaign.

.. raw:: html

    <script src="https://donorbox.org/widget.js" paypalExpress="false"></script><iframe src="https://donorbox.org/embed/formulas?amount=25&show_content=true" height="685px" width="100%" style="max-width:100%; min-width:100%; max-height:none!important" seamless="seamless" name="donorbox" frameborder="0" scrolling="no" allowpaymentrequest></iframe>


.. note::

    The cumulative donation amount per function is calculated as the example:

    ======== ========= ========= ========= ====== ======================
    Function Donator 1 Donator 2 Donator 3  TOT    Implementation order
    -------- --------- --------- --------- ------ ----------------------
       -        150€      120€      50€     -            -
    ======== ========= ========= ========= ====== ======================
      SUM       50€       40€       25€     125€         1st
      SIN       50€                 25€     75€          3rd
      TAN       50€       40€               90€          2nd
      COS                 40€               40€          4th
    ======== ========= ========= ========= ====== ======================
