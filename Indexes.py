#!/usr/bin/python
import numpy

class Indexes:
    """
    Indexes object generates indices for different configurations
    Modes:
    'r' : random
    'd' : deterministic
    'k' : k-fold cross-fold validation
    """
    def __init__(self):
        self.mode = 'r'
        self.iterations = 10
        __init__('r',10,300)
    def __init__(self,mode,iterations,train_size):
        self.mode = mode
        self.iterations = iterations
        self.train_size = train_size
        self.pos_train_ind = None 
        self.pos_test_ind = None
        self.neg_train_ind = None
        self.neg_test_ind = None
        self.gen_indices = generate_indices(mode,iterations,train_size)
    def next(self):
        (a,b,c,d) = self.gen_indices.next()
        self.pos_train_ind = a 
        self.pos_test_ind = b
        self.neg_train_ind = c
        self.neg_test_ind = d
    def get_pos_train_ind(self):
        return self.pos_train_ind
    def get_pos_test_ind(self):
        return self.pos_test_ind
    def get_neg_train_ind(self):
        return self.neg_train_ind
    def get_neg_test_ind(self):
        return self.neg_test_ind

def shuffle_ind():
    ind = numpy.arange(1000)
    from numpy.random import shuffle
    shuffle(ind)
    return ind

def generate_indices(mode='r',iterations=1,train_size=300):
    if mode=='d': # deterministic
        def get_indices():
            ind = numpy.arange(1000)
            pos_train_ind = ind[:train_size]
            pos_test_ind = ind[train_size:]
            neg_train_ind = ind[:train_size]
            neg_test_ind = ind[train_size:]
            for i in range(iterations):
                yield (pos_train_ind, pos_test_ind, neg_train_ind, neg_test_ind)
    elif mode=='r': # random
        def get_indices():
            for i in range(iterations):
                pos_ind = shuffle_ind()
                pos_train_ind = pos_ind[:train_size]
                pos_test_ind = pos_ind[train_size:]
                neg_ind = shuffle_ind()
                neg_train_ind = neg_ind[:train_size]
                neg_test_ind = neg_ind[train_size:]
                yield (pos_train_ind, pos_test_ind, neg_train_ind, neg_test_ind)
    elif mode=='k': # k-fold cross-validation
        # here, iterations = number of folds
        pos_ind = shuffle_ind()
        neg_ind = shuffle_ind()
        pos_folds = numpy.array_split(pos_ind,iterations)
        neg_folds = numpy.array_split(neg_ind,iterations)
        def get_indices():
            for i in range(iterations):
                pos_train_ind = numpy.hstack(pos_folds[:i] + pos_folds[i+1:]).tolist()
                pos_test_ind = pos_folds[i].tolist() 
                neg_train_ind = numpy.hstack(neg_folds[:i] + neg_folds[i+1:]).tolist()
                neg_test_ind = neg_folds[i].tolist() 
                yield (pos_train_ind, pos_test_ind, neg_train_ind, neg_test_ind)
    return get_indices()
