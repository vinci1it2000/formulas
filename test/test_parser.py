
import unittest
from formulas.parser import Parser
from formulas.errors import FormulaError
import ddt
from formulas.constants import NAME_REFERENCES
import schedula.utils as sh_utl


@ddt.ddt
class TestParser(unittest.TestCase):
    @ddt.data(
        ('=L4:N15 : J5:L12', 'L4:N15:J5:L12'),
        ('=L4:N15 : c', 'L4:N15:c'),
        ('=ciao:bau', 'ciao:bau'),
        ('=SUM(L4:N15 (J5:L12, J5:L12, N5:P12, J5:L12, J5:L12))',
         'SUM(L4:N15 (J5:L12,J5:L12,N5:P12,J5:L12,J5:L12))'),
        ('=SUM(L4:N15 (J5:L12, N5:P12))', 'SUM(L4:N15 (J5:L12,N5:P12))'),
        ('=(-INT(2))', '(u-INT(2))'),
        ('=(1+1)+(1+1)', '(1+1)+(1+1)'),
        ('=( 1 + 2 + 3)*(4 + 5)^(1/5)', '(1+2+3)*(4+5)^(1/5)'),
        ('={1,2;1,2}', 'ARRAY(ARRAY(1,2)ARRAY(1,2))'),
        ('=PI()', 'PI()'),
        ('=INT(1)%+3', 'INT(1)%+3'),
        ('= 1 + 2 + 3 * 4 ^ 5 * SUM(a, 1+b)', '1+2+3*4^5*SUM(a,1+b)'),
        ('=SUM({1})', 'SUM(ARRAY(ARRAY(1)))'),
        ('= a b', 'a b'),
        ('="  a"', '  a'),
        ('=#NULL!', '#NULL!'),
        ('=1 + 2', '1+2'),
        ('{=a1:b1 + INDIRECT("A1:B2")}', 'a1:b1+A1:B2'),
        ('= a + IF(a, b, c)', 'a+IF(a,b,c)'),
        ('=AVERAGE(((123 + 4 + AVERAGE(A1:A2))))',
         'AVERAGE(((123+4+AVERAGE(A1:A2))))'),
        ('=a,b', 'a,b'),
        ('=a b', 'a b'))
    def test_valid_formula(self, case):
        inputs, result = case
        tokens, ast = Parser().ast(inputs)
        output = ''.join(t.name for t in tokens)
        self.assertEqual(result, output, '%s != %s' % (result, output))

    @ddt.data(
        '=-', '={}', '=1  -*  4', '={1;2,2}', '= a + IF((a, b, c)',
        '= a + IF(}a, b, c)', '1+2', '=a 1', '=a INT(1)')
    def test_invalid_formulas(self, inputs):
        with self.assertRaises(FormulaError):
            Parser().ast(inputs)

    @ddt.data(
        ('=(L4:N7 (K5:L6, N5:O6))', (
            [(2,), (2,)],
            [(3,), (3,)]
        ), '[3 3 2 2]'),
        ('=(a (b, c))', (
            [(1, 1, 1),
             (2, 1, 3),
             (2, 1, 3),
             (1, 1, 1)],
            [(3, 3), (3, 3)],
            [(2, 2), (2, 2)],
            {'a': 'L4:N7', 'b': 'K5:L6', 'c': 'N5:O6'}
        ),
         '[3 3 2 2]'),
        ('=(-INT(2))', (), '-2'),
        ('=(1+1)+(1+1)', (), '4'),
        ('=( 1 + 2 + 3)*(4 + 5)^(1/5)', (), '9.311073443492159'),
        ('={1,2;1,2}', (), '[[1 2]\n [1 2]]'),
        ('=PI()', (), '3.141592653589793'),
        ('=INT(1)%+3', (), '3.01'),
        ('=SUM({1, 3; 4, 2})', (), '10'),
        ('=" "" a"', (), ' " a'),
        ('=#NULL!',  (), 'NULL!'),
        ('=1 + 2', (), '3'),
        ('=AVERAGE(((123 + 4 + AVERAGE({1,2}))))', (), '128.5'),
        ('="a" & "b"""', (), 'ab"'))
    def test_compile(self, case):
        formula, inputs, result = case
        func = Parser().ast(formula)[1].compile()
        output = str(func(*inputs))
        self.assertEqual(result, output, '{} != {}'.format(result, output))
