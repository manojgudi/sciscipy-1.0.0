import unittest
from sciscipy import read, eval
import numpy

try:
    import numpy
    numpy_is_avail = 1
except ImportError:
    numpy_is_avail = 0
	
class test_read(unittest.TestCase):
    def setUp(self):
        pass

    def test_read_tlist(self):
        eval("x=tlist(['test','a','b'],12,'item')")
        x=read('x')
        if numpy_is_avail:
            num = numpy.array(12)
        else:
            num = 12
        py_x = {'__tlist_name': 'test', 'a': num, 'b': ['item']}
        assert x == py_x, str(py_x) + " != tlist(['test','a','b'],12,'item')"