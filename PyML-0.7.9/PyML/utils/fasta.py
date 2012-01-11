
# Copyright (C) 2003, 2004 by BiRC -- Bioinformatics Research Center
#                                     University of Aarhus, Denmark
#                                     Contact: Thomas Mailund <mailund@birc.dk>
# 
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or (at
# your option) any later version.
# 
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307,
# USA.

"""
A parser for FASTA files.

Copyright (C) 2003, 2004 by BiRC -- Bioinformatics Research Center
                                    University of Aarhus, Denmark
                                    Contact: Thomas Mailund <mailund@birc.dk>
with changes by Asa Ben-Hur
"""

from __future__ import generators
import os


def myopen(fileName) :

    if not ( os.path.exists(fileName) and os.path.isfile(fileName) ):
        raise ValueError, 'file does not exist at %s' % fileName
    
    import gzip
    fileHandle = gzip.GzipFile(fileName)

    gzippedFile = True
    try :
        line = fileHandle.readline()
        fileHandle.close()
    except :
        gzippedFile = False

    if gzippedFile :
        return gzip.GzipFile(fileName)
    else :
        return open(fileName)


class MalformedInput :
    "Exception raised when the input file does not look like a fasta file."
    pass

class FastaRecord :
    "a fasta record."

    def __init__(self, header, sequence):
        "Create a record with the given header and sequence."
        self.header = header
        self.sequence = sequence

    def __str__(self) :

        return '>' + self.header + '\n' + self.sequence + '\n'
        

def _fasta_itr_from_file(file) :
    "Provide an iteration through the fasta records in file."

    h = file.readline()[:-1]
    if h[0] != '>':
        raise MalformedInput()
    h = h[1:]

    seq = []
    for line in file:
        line = line[:-1] # remove newline

        if line[0] == '>':
            yield FastaRecord(h,''.join(seq))

            h = line[1:]
            seq = []
            continue

        #seq += [line]
        seq.append(line)

    yield FastaRecord(h,''.join(seq))


def _fasta_itr_from_name(fname):
    "Provide an iteration through the fasta records in the file named fname. "

    f = myopen(fname)
    for rec in _fasta_itr_from_file(f) :
        yield rec


def _fasta_itr(src):
    """Provide an iteration through the fasta records in file `src'.
    
    Here `src' can be either a file object or the name of a file.
    """
    if type(src) == str :
        return _fasta_itr_from_name(src)
    elif type(src) == file :
        return _fasta_itr_from_file(src)
    else:
        raise TypeError

def fasta_get_by_name(itr,name):
    "Return the record in itr with the given name."
    x = name.strip()
    for rec in itr:
        if rec.header.strip() == x:
            return rec
    return None

class fasta_itr (object) :
    "An iterator through a sequence of fasta records."

    def __init__(self, src) :
        "Create an iterator through the records in src."

        self.__itr = _fasta_itr(src)

    def __iter__(self) :

        return self

    def next(self) :

        return self.__itr.next()

    def __getitem__(self,name) :

        return fasta_get_by_name(iter(self),name)

class fasta_slice (object) :

    """Provide an iterator through the fasta records from
    'start' to 'stop'.

    """
    def __init__(self, src, first, last = None):
        """
        :Parameters:
        - `src` - the fasta file/file handle. file can be gzipped.
        - `first` - the first record (either its index in the file or
          its identifier
        - `last` - the last record to be output (index in the file or identifier)
        """
        self.__itr = _fasta_itr(src)
        self.__first = first
        self.__last = last
        if type(first) == int :
            self.__current = 0
        elif type(first) == type('') :
            self.__current = None
        else :
            raise ValueError, 'bad first'

        self.__foundFirst = False
        if self.__first == 0 or self.__first == '' :
            self.__foundFirst = True


    def __iter__(self) :

        return self

    def next(self) :

        if not self.__foundFirst :
            for rec in self.__itr :
                if type(self.__first) == int :
                    if self.__first == self.__current :
                        self.__foundFirst = True
                        break
                    self.__current += 1
                else :
                    if rec.header == self.__first :
                        self.__foundFirst = True
                        break
                    self.__current = rec.header
            if not self.__foundFirst :
                raise ValueError, 'did not find first record'
            return rec

        rec = self.__itr.next()

        if self.__last is not None :
            if type(self.__first) == int :
                self.__current += 1
                if self.__current == self.__last :
                    raise StopIteration
            else :
                if rec.header == self.__last :
                    raise StopIteration
                self.__current = rec.header

        return rec


    def __getitem__(self, name):

        return fasta_get_by_name(iter(self),name)

    def save(self, fileName) :

        outfile = open(fileName, 'w')
        for record in self :
            outfile.write(str(record))

def get_sequence(src, name):
    "Return the record in src with the given name."

    return fasta_itr(src)[name]


def fasta_count(src) :
    """
    count the number of records in a fasta file
    """

    num_records = 0
    for rec in fasta_itr(src) :
        num_records += 1

    return num_records


def fasta_split(fileName, num_files, directory = None) :
    """
    split a fasta file into a given number of files
    the resulting files are named by adding a number to the provided file name.

    :Parameters:
    - `fileName` - the fasta file to split
    - `num_files` - the number of files to split into
    - `directory` - the directory into which to write the files
    """

    num_records = fasta_count(fileName)
    print num_records
    if directory is None :
        base, ext = os.path.splitext(fileName)
    else :
        dir, name = os.path.split(fileName)
        base, ext = os.path.splitext(name)
        base = os.path.join(directory, base)
    print base
    rec_num = 0
    file_num = 1
    recs_per_file = num_records / num_files + 1
    for rec in fasta_itr(fileName) :
        if rec_num % recs_per_file == 0 :
            outfile = open(base + '.' + str(file_num) + ext, 'w')
            file_num += 1
        outfile.write(str(rec))
        rec_num += 1
    
def fasta_sample(infile, outfile, size = None) :

    import random
    print outfile
    numseq = fasta_count(infile)
    if size is None : size = numseq
    seqs = range(numseq)
    random.shuffle(seqs)
    seqs = dict.fromkeys(seqs[:size])
    if type(outfile) == type('') :
        outfile = open(outfile, 'w')
    elif not hasattr(outfile, 'write') :
        raise ValueError
    i = 0
    for rec in fasta_itr(infile) :
        if i in seqs :
            outfile.write('>' + rec.header + '\n' + rec.sequence + '\n')
        i += 1
    
def fasta_shuffle(infile, outfile = None) :

    import tempfile
    import random
    if outfile is None :
        fid, outfile_name = tempfile.mkstemp()
    else :
        outfile_name = outfile
    print 'outfile', outfile_name
    outfile_handle = open(outfile_name, 'w')

    recs = [rec for rec in fasta_itr(infile)]
    numseq = len(recs)
    seqs = range(numseq)
    random.shuffle(seqs)
    for idx in seqs :
        outfile_handle.write('>' + recs[idx].header + '\n' + recs[idx].sequence + '\n')
    outfile_handle.close()

    if outfile is None :
        os.rename(outfile_name, infile)


def fasta_subset(infileName, outfileName, ids) :

    if type(ids) != type({}) :
        import misc
        ids = misc.list2dict(ids)

    outfile = open(outfileName, 'w')
    for rec in fasta_itr(infileName) :
        if rec.header in ids :
            outfile.write(str(rec))

def fasta_delimiter(fastaFile) :

    rec = fasta_itr(fastaFile).next()
    if rec.header.find('|') >= 0 :
        return '|'
    else :
        return None


if __name__ == '__main__':

    import sys
    if len(sys.argv) != 2:
        print "missing file name"
        sys.exit(2)

    print 'iterating through all sequences in input file'
    for rec in fasta_itr(sys.argv[1]):
        print rec

    print 'iterating through input, from the second sequence'
    for rec in fasta_slice(sys.argv[1], 1, 3):
        print rec
