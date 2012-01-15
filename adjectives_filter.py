#!/usr/bin/python
import os

POSPOS_DIR = "pos_tagged"
NEGPOS_DIR = "neg_tagged"
POSADJ_DIR = "pos_adj"
NEGADJ_DIR = "neg_adj"

def filter_adj(olddir, newdir):
    if not os.path.isdir(newdir):
        os.mkdir(newdir)
    for filename in os.listdir(olddir):
        f = open("%s/%s" % (olddir,filename)).read().split("\n")
        w = open("%s/%s" % (newdir,filename), 'w')
        for word in f:
            if word[-3:]=='_JJ':
                w.write("%s\n" % word)
        w.close()

if __name__ == "__main__":
    filter_adj(POSPOS_DIR,POSADJ_DIR)
    filter_adj(NEGPOS_DIR,NEGADJ_DIR)
