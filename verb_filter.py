#!/usr/bin/python
import os

def filter_adj(olddir, newdir):
    if not os.path.isdir(newdir):
        os.mkdir(newdir)
    for filename in os.listdir(olddir):
        f = open("%s/%s" % (olddir,filename)).read().split("\n")
        w = open("%s/%s" % (newdir,filename), 'w')
        for word in f:
            if word[-4:]=='_VBZ' or word[-4:]=="_VBD" or word[-3:]=="_VB":
                w.write("%s\n" % word)
        w.close()

if __name__ == "__main__":
    # usage: python adjectives_filter.py -d neg
    # usage: python adjectives_filter.py -d yelp/default/1star
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-d", "--dir", dest="directory")
    (options, args) = parser.parse_args()

    olddir = "%s_tagged" % options.directory
    newdir = "%s_verb" % options.directory
    filter_adj(olddir,newdir)
