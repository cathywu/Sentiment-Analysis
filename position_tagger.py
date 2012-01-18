#!/usr/bin/python
import os

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
    # usage: python position_tagger.py -d neg
    # usage: python position_tagger.py -d yelp/default/1star
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-d", "--dir", dest="directory")
    (options, args) = parser.parse_args()

    olddir = options.directory
    newdir = "%s_position" % olddir
    tagfiles(olddir, newdir)
