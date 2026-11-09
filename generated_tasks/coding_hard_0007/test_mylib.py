import unittest
import mylib
class TestOps(unittest.TestCase):
    def test_rotate(self):
        self.assertEqual(mylib.rotate([1,2,3,4], 1), [4,1,2,3])
        self.assertEqual(mylib.rotate([1,2,3], -1), [2,3,1])
    def test_empty(self):
        self.assertEqual(mylib.rotate([], 5), [])
    def test_palindrome(self):
        self.assertTrue(mylib.is_palindrome('A man, a plan, a canal: Panama!'))
        self.assertFalse(mylib.is_palindrome('hello'))
    def test_flatten(self):
        self.assertEqual(mylib.flatten_once([[1,2],[3],[],[4,5]]), [1,2,3,4,5])
if __name__ == '__main__': unittest.main()
