import unittest
import sciscipy as sci

try:
    import numpy
    numpy_is_avail = 1
except ImportError:
    numpy_is_avail = 0

class test_matrix(unittest.TestCase):
    def setUp(self):
        pass

    def test_readwrite1d(self):
        sci.eval("x=rand(1, 100)")
        sci.eval("my_sum=sum(x)")
        x = sci.read("x")
        my_sum = 0
        for i in range(len(x)):
            my_sum += x[i]
        
        my_other_sum = sci.read("my_sum")
        assert(my_other_sum[0] == my_sum)

    def test_readwrite1dT(self):
        sci.eval("x=rand(100, 1)")
        sci.eval("my_sum=sum(x)")
        x = sci.read("x")
        my_sum = 0
        for i in range(len(x)):
            my_sum += x[i]
        
        my_other_sum = sci.read("my_sum")
        assert(my_other_sum[0] == my_sum)        

    def test_readwrite(self):              
        sci.eval("x=[1,2,3 ; 4,5,6]")
        y = sci.read("x")
        sci.write("z", y)
        w = sci.read("z")
        if numpy_is_avail:
            assert(numpy.alltrue(numpy.equal(y, w)))
        else:
            assert(y == w)

    def test_bigmat(self):
        sci.eval("x=rand(1000,800)")
        y = sci.read("x")
        sci.write("xx", y)
        sci.eval("dist = sum((x - xx).^2)")
        dist = sci.read("dist")
        assert(dist[0] == 0)

    def test_complex(self):
        sci.eval("x=[1+11*%i, 2+222*%i, 3+333*%i ; 4+444*%i , 5+55*%i, 6+66*%i]")
        y = sci.read("x")
        sci.write("z", y)
        w = sci.read("z")
        
        if numpy_is_avail:
            assert(numpy.alltrue(numpy.equal(y, w)))
        else:
            assert(y == w)


if __name__ == '__main__':
    unittest.main()




