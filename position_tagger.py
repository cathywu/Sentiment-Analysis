#!/usr/bin/python
import os

POS_DIR = "pos"
NEG_DIR = "neg"
POSTAGGED_DIR = "pos_position"
NEGTAGGED_DIR = "neg_position"

def tagfiles(olddir, newdir):
    if not os.path.isdir(newdir):
        os.mkdir(newdir)
    for filename in os.listdir(olddir):
        f = open("%s/%s" % (olddir,filename)).read().split(" ")
        w = open("%s/%s" % (newdir,filename), 'w')
        length = len(f)
        for i in range(length):
            if i < length/4:
                w.write("%s_Q1 " % f[i])
            elif i < length*3/4:
                w.write("%s_Q23 " % f[i])
            else: 
                w.write("%s_Q4 " % f[i])
        w.close()

if __name__ == "__main__":
    tagfiles(POS_DIR,POSTAGGED_DIR)
    tagfiles(NEG_DIR,NEGTAGGED_DIR)
