import unittest
import mylib

class TestLibrary(unittest.TestCase):
    def test_clamp(self):
        self.assertEqual(mylib.clamp(12, 0, 10), 10)
        self.assertEqual(mylib.clamp(-2, 0, 10), 0)
        self.assertEqual(mylib.clamp(4, 0, 10), 4)
    def test_unique(self):
        self.assertEqual(mylib.unique_in_order([3, 1, 3, 2, 1]), [3, 1, 2])
    def test_median_odd(self):
        self.assertEqual(mylib.median([9, 1, 5]), 5)
    def test_median_even(self):
        self.assertEqual(mylib.median([1, 8, 3, 4]), 3.5)

if __name__ == '__main__':
    unittest.main()
