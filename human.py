#!/usr/bin/python

import random
from random import randrange
import numpy
from numpy.random import shuffle
import os

tests = 10 # number of test subjects
n = 6 # number of reviews per test subject

def random_insert(lst, item):
    lst.insert(randrange(len(lst)+1), item)

def get_id(cls, size=6, chars="QWERTYIOPASDFGHJKZXCVBNM"):
    if cls == 1:
        # L marks positive
        special="L"
    else:
        # U marks negative
        special="U"
    lst = [random.choice(chars) for x in range(size)]
    random_insert(lst, special)
    return ''.join(lst)

rev = numpy.arange(2000)
shuffle(rev)
print rev

w = open("human.txt",'w')
pos_files = os.listdir("pos")
neg_files = os.listdir("neg")

for r in rev:
    if r >= 1000:
        r = r - 1000
        directory = "pos"
        tag = get_id(1)
        f = open("%s/%s" % (directory,pos_files[r])).read()
    else:
        directory = "neg"
        tag = get_id(0)
        f = open("%s/%s" % (directory,neg_files[r])).read()

    w.write(f)
    w.write("\n\n%s%s===========================================================================\n\n" % (tag,r))
w.close()




