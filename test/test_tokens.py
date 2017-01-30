
import unittest
from formulas.tokens.operand import String, Error, Range
from formulas.tokens.operator import OperatorToken
from formulas.errors import TokenError
import ddt


@ddt.ddt
class TestTokens(unittest.TestCase):
    @ddt.data(('A:A', 'A:A'), ('1:1', '1:1'))
    def test_range(self, case):
        inputs, result = case
        output = Range(inputs).name
        self.assertEqual(result, output, '%s != %s' % (result, output))

    @ddt.data(
        (' <=', '<='), (' <>', '<>'), ('  <', '<'), ('>', '>'), ('>=', '>='),
        ('=', '='), ('++', '+'), ('---', '-'), ('++--+', '+'), (' *', '*'),
        ('^ ', '^'), (' & ', '&'), ('/', '/'), ('%', '%'), (' : ', ':'))
    def test_valid_operators(self, case):
        inputs, result = case
        output = OperatorToken(inputs).name
        self.assertEqual(result, output, '%s != %s' % (result, output))

    @ddt.data(
        '=<', '> <', ' < <', '>>', '=>', '==', '+*', '**', '^ ^', 'z&', '\/',
        '', '%%', ', ,', ' : : ')
    def test_invalid_operators(self, inputs):
        with self.assertRaises(TokenError):
            OperatorToken(inputs)

    @ddt.data(
        ('""', ''), (' " "', ' '), ('  "a\'b"', 'a\'b'), ('" a " b"', ' a '),
        ('" "" "', ' "" '))
    def test_valid_strings(self, case):
        inputs, result = case
        output = String(inputs).name
        self.assertEqual(result, output, '%s != %s' % (result, output))

    @ddt.data('"', '" ', 'a\'b"', ' a " b')
    def test_invalid_strings(self, inputs):
        with self.assertRaises(TokenError):
            String(inputs)

    @ddt.data(
        ('#NULL! ', '#NULL!'), ('  #DIV/0!', '#DIV/0!'), ('#VALUE!', '#VALUE!'),
        ('#REF!', '#REF!'), ('#NAME?', '#NAME?'), ('#NUM!', '#NUM!'),
        ('#N/A  ', '#N/A'))
    def test_valid_errors(self, case):
        inputs, result = case
        output = Error(inputs).name
        self.assertEqual(result, output, '%s != %s' % (result, output))

    @ddt.data(
        '#NUL L!', ' #DIV\/0!', '# VALUE!', '#REF', '#NME?', '#NUM !', '#N\A ')
    def test_invalid_strings(self, inputs):
        with self.assertRaises(TokenError):
            Error(inputs)
