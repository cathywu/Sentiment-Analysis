from numpy import *
from scipy.sparse import lil_matrix, csr_matrix, csc_matrix

"""
pass in data as a numpy matrix
with data in vectors, class as last row
or as dict mapping from point -> class
"""
class DefDict(dict):
    def __init__(self, default, *args, **kwargs):
        self.default = default
        self.update(*args, **kwargs)
    def __getitem__(self, key):
        if key not in self:
            return self.default
        return dict.__getitem__(self, key)
    def __setitem__(self, key, val):
        dict.__setitem__(self, key, val)
    def update(self, *args, **kwargs):
        for other in args:
            for k in other:
                self[k] = other[k]
    def addto(self, other):
        for key in other:
            self[key] += other[key]
                    
class Data:
    def __init__(self, inp):
        self.matrix = None
        self.dict = None
        self.sparse = None
        if type(inp) == ndarray: 
            self.matrix = inp
        elif type(inp) == DefDict:
            self.dict = inp
        elif type(inp) == csc_matrix or type(inp) == csr_matrix or type(inp) == lil_matrix:
            self.sparse = inp
        else:
            print type(inp)
            raise RuntimeError
            
    def asDict(self):
        if self.dict == None:
            if self.matrix == None:
                raise RuntimeError
            self.dict = DefDict([])
            for col in self.matrix.T:
                print "col:", col, col[:-1], col[-1], self.dict[tuple(col[:-1])]
                self.dict[tuple(col[:-1])] = \
                    tuple(self.dict[tuple(col[:-1])]) + (int(col[-1]),)
        return self.dict

    def asMatrix(self):
        if self.matrix == None:
            if not self.sparse == None:
                self.matrix = array(self.sparse)
            else:
                cols = []
                for k in self.dict:
                    for v in self.dict[k]:
                        cols.append(hstack((k, v)).T)
                        self.matrix = column_stack(cols)
        return self.matrix

    def asSparseMatrix(self):
        if self.sparse == None:
            raise RuntimeError
        return self.sparse
            

if __name__ == "__main__":
    testmat = DefDict((), {(1,2,3,4,5):(1,),
                       (4,2,5,1,0):(2,),
                       (5,3,2,1,1):(3,4)})
    data1 = Data(testmat)
    data2 = Data(data1.asMatrix())
    print (data1.asMatrix() == data2.asMatrix()).all()
    print data2.asDict() == data1.asDict()

