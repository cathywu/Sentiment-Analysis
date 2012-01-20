#!/usr/bin/python
import os
from numpy.random import permutation

def select_files(olddir, newdir, n=1000):
    spaceout = '.,()"?!:;[]{}|*&^%$#'

    if not os.path.isdir(newdir):
        os.mkdir(newdir)
    files = os.listdir(olddir)
    nfiles = len(files)
    revs = permutation(nfiles)[:n]

    for rev in revs:
        f = open("%s/%s" % (olddir,files[rev])).read().lower()
        out = ""
        for char in f:
            if char in spaceout:
                out += " %s " % char
            else:
                out += char
        out = out.replace("  "," ").replace("  "," ")
        w = open("%s/%s" % (newdir,files[rev]), 'w')
        w. write(out)
        w.close()
        
if __name__ == "__main__":
    # usage: python position_tagger.py -d neg
    # usage: python position_tagger.py -d yelp/default/1star
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-d", "--dir", dest="directory")
    (options, args) = parser.parse_args()

    olddir = options.directory
    newdir = "%s_limited" % olddir
    select_files(olddir, newdir)
