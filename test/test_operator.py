
import unittest
from formulas.formulas.operators import Ranges
import ddt


@ddt.ddt
class TestTokens(unittest.TestCase):
    @ddt.data(
        ((('D7:F14',), ('C:E',)), '<Ranges>(D7:F14, C:C, D:E6, D15:E)'),
        ((('D7:F14',), ('C4:E9',)), '<Ranges>(D7:F14, C4:C9, D4:E6)'),
        ((('I7:L14',), ('H9:M12',)), '<Ranges>(I7:L14, H9:H12, M9:M12)'),
        ((('I7:L14',), ('J5:K16',)), '<Ranges>(I7:L14, J5:K6, J15:K16)'),
        ((('F24:I32',), ('G20:H26',)), '<Ranges>(F24:I32, G20:H23)'),
        ((('M23:P30',), ('L20:Q25',)),
         '<Ranges>(M23:P30, L20:L25, Q20:Q25, M20:P22)'),
        ((('M23:P30', 'L20:L25', 'Q20:Q25', 'M20:P22',), ('L20:Q25',)),
         '<Ranges>(M23:P30, L20:L25, Q20:Q25, M20:P22)'),
        ((('M23:P30', 'L20:L25', 'Q20:Q25', 'M20:P22',),
          ('P18:R27', 'L25:N32')),
         '<Ranges>(M23:P30, L20:L25, Q20:Q25, M20:P22, R18:R27, Q18:Q19,'
         ' Q26:Q27, P18:P19, L26:L32, M31:N32)'))
    def test_union_ranges(self, case):
        (range1, range2), result = case
        output = str(Ranges().push(*range1) + Ranges().push(*range2))
        self.assertEqual(result, output, '%s != %s' % (result, output))

    @ddt.data(
        ((('D7:F14',), ('C:E',)), '<Ranges>(D7:E14)'),
        ((('D7:F14',), ('C4:E9',)), '<Ranges>(D7:E9)'),
        ((('I7:L14',), ('H9:M12',)), '<Ranges>(I9:L12)'),
        ((('I7:L14',), ('J5:K16',)), '<Ranges>(J7:K14)'),
        ((('F24:I32',), ('G20:H26',)), '<Ranges>(G24:H26)'),
        ((('M23:P30',), ('L20:Q25',)), '<Ranges>(M23:P25)'),
        ((('M23:P30', 'L20:L25', 'Q20:Q25', 'M20:P22',), ('L20:Q25',)),
         '<Ranges>(M23:P25, L20:L25, Q20:Q25, M20:P22)'),
        ((('M23:P30', 'L20:L25', 'Q20:Q25', 'M20:P22',),
          ('P18:R27', 'K25:N32')),
         '<Ranges>(P23:P27, Q20:Q25, P20:P22, M25:N30, L25)'))
    def test_intersection_ranges(self, case):
        (range1, range2), result = case
        output = str(Ranges().push(*range1) - Ranges().push(*range2))
        self.assertEqual(result, output, '%s != %s' % (result, output))

    @ddt.data(
        (('D7:F14', 'C:C', 'D:E6', 'D15:E'),
         '<Ranges>(C:C, D:D15, E:E15, F7:F14)'))
    def test_simplify_ranges(self, case):
        ranges, result = case
        output = str(Ranges().push(*ranges).simplify())
        self.assertEqual(result, output, '%s != %s' % (result, output))
