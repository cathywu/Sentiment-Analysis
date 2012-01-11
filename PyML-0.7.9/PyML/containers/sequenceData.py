
from PyML.containers.labels import Labels
#from PyML.containers.ext import cpositionalkmerdata
from PyML.containers.ext import csequencedata
from PyML.containers.baseDatasets import WrapperDataSet, BaseDataSet
from PyML.containers.vectorDatasets import SparseDataSet
from PyML.utils import fasta

class SequenceBase (BaseDataSet) :

    def copy(self, other, patterns, deepcopy) :
        if patterns is None :
            patterns = range(len(other))
        self.container.__init__(self, other, patterns)

    def constructFromFile(self, fileName, **args) :
        print 'reading from', fileName
        headerHandler = fastaHeaderHandler
        if 'headerHandler' in args :
            headerHandler = args['headerHandler']
        numPatterns = fasta.fasta_count(fileName)
        self.container.__init__(self, numPatterns)

        patternIDs = []
        L = []
        for record in fasta.fasta_itr(fileName) :
            self.addPattern(record.sequence)
            patternID, label = headerHandler(record.header)
            patternIDs.append(patternID)
            if label is not None :
                L.append(label)

        self.attachLabels(Labels(L, patternID = patternIDs, **args))

    def fromArray(self, X, **args) :
        self.container.__init__(self, len(X))
        for x in X :
            self.addPattern(x)

    def __len__(self) :

        return self.size()

    def save(self, fileName) :
        fileHandle = open(fileName, 'w')
        for seqid in range(len(self)) :
            fileHandle.write('>' + self.labels.patternID[seqid] + '\n')
            seq = self.getSequence(seqid)
            fileHandle.write(seq + '\n')
    

class SequenceData (WrapperDataSet, csequencedata.SequenceData, SequenceBase) :
    
    def __init__(self, arg=None, **args) :
        self.container = csequencedata.SequenceData
        BaseDataSet.__init__(self, arg, **args)
        self.initialize(**args)
       
    def initialize(self, **args) :
        values = {'mink' : 2,
                  'maxk' : 2,
                  'mismatches' : 0,
                  'mismatchProfile' : [0,0,1,1,1,1,2,2,3,3,3,3],
                  'maxShift' : 0,
                  'noShiftStart' : 0,
                  'noShiftEnd' : 0}
        values.update(args)
        if len(values['mismatchProfile']) < values['maxk'] and values['mismatches'] > 0 :
            raise ValueError, 'mismatchProfile not long enough'
        # if no mismatches are allowed the mismatch profile needs to be all 0
        if values['mismatches'] == 0 :
            values['mismatchProfile'] = [0 for i in range(values['maxk'])]
        self.setMismatchProfile(values['mismatchProfile'])
        shiftWeight = [1.0 / (2 * (abs(s) + 1)) for s in range(values['maxShift'] + 1)]
        self.setShiftWeight(shiftWeight)
        self.mink = values['mink']
        self.maxk = values['maxk']
        self.mismatches = values['mismatches']
        self.maxShift = values['maxShift']
        self.noShiftStart = values['noShiftStart']
        self.noShiftEnd = values['noShiftEnd']
 
 
def fastaHeaderHandler(header) :
    return header.split()[0], None

def fasta_read(file_name) :
    """read the sequence from a file in fasta format"""
    return [record.sequence for record in fasta.fasta_itr(file_name)]

def generate_spectrum(sequences, k, prefix = '', skip = []) :
    data = []
    for s in sequences :
        kmers = {}
        for i in range(len(s) - k + 1) :
            kmer = s[i:i+k]
            skip_flag = False
            if len(skip) > 0 :
                for c in kmer :
                    if c in skip : skip_flag = True
            if skip_flag :
                continue
            if len(prefix) > 0 : 
                kmer = prefix + kmer
            if kmer not in kmers :
                kmers[kmer] = 0
            kmers[kmer] += 1.0
        data.append(kmers)
    return data

def generate_single_mismatch_spectrum(sequences, k, prefix='', skip=[]) :
    spectrum_data = []
    for seq in sequences :
        kmers = {}
        for i in range(len(seq) - k + 1) :
            kmer = seq[i:i + k]
            for i in range(0, len(kmer) - 1) :
                if i > 0 :
                    mismatch_kmer = kmer[:i] + '.' + kmer[i + 1:]
                else :
                    # the case of matching with no mismatches
                    mismatch_kmer = kmer
                skip_flag = False
                if len(skip) > 0 :
                    for c in kmer :
                        if c in skip : skip_flag = True
                if skip_flag :
                    continue
                if len(prefix) > 0 : 
                    kmer = prefix + kmer
                if mismatch_kmer not in kmers :
                    kmers[mismatch_kmer] = 0
                kmers[mismatch_kmer] += 1.0
        spectrum_data.append(kmers)
    return spectrum_data

def spectrum_data(sequences, k1, k2=None, **args) :
    """generate a dataset object that represents the spectrum of a sequence,
    i.e. its kernel function is the spectrum kernel.
    Reference:
    C. Leslie, E. Eskin, and WS Noble. 
    The spectrum kernel:  A string kernel for SVM protein classification.

    :Parameters:
      - `sequences` - either a name of a fasta file that contains the sequences or
         a list of sequences
      - `k1` - the length of the substrings to consider
      - `k2` - if k2 is provided then strings whose length is between k1
        and k2 are used in constructing the spectrum.

    :Keywords:
      - `normalize` - whether to normalize the dataset [default: True]
      - `prefix` - a string to be added to the name of each feature 
        (useful when combining spectrum features from several sources)
      - `skip` - a list of characters that should be skipped in computing
        the spectrum [default: []].  Whenever a character in this list is
        encountered, the substring is not included in the spectrum
      - `mismatch` - whether to allow a single mismatch [default: False]
    """
    prefix = ''
    if 'prefix' in args :
        prefix = args['prefix']
    normalize = True
    if 'normalize' in args :
        normalize = args['normalize']
    if k2 is None : 
        k2 = k1 + 1
    skip = []
    if 'skip' in args :
        skip = args['skip']
    mismatch = False
    if 'mismatch' in args :
        mismatch = args['mismatch']
    if mismatch :
        spectrum_generator = generate_single_mismatch_spectrum
    else :
        spectrum_generator = generate_spectrum
    if type(sequences) == type('') :
        sequences = fasta_read(sequences)

    data = SparseDataSet(spectrum_generator(sequences, k1, prefix, skip))
    if normalize :
        data.normalize(2)
    for k in range(k1+1, k2 + 1) :
        data2 = SparseDataSet(spectrum_generator(sequences, k, prefix, skip))
        if normalize :
            data2.normalize(2)
        data.addFeatures(data2)

    return data

def generate_gappy_pairs(sequences, maximum_distance, prefix='', skip=[]) :
    data = []
    for seq in sequences :
        motif_composition = {}
        for pos1 in range(len(seq) - 1) :
            for offset in range(1, min(maximum_distance, len(seq) - pos1 - 1) + 1) :
                motif = prefix + seq[pos1] + seq[pos1 + offset] + str(offset)
                if motif[0] in skip or motif[1] in skip :
                    continue
                if motif not in motif_composition :
                    motif_composition[motif] = 0
                motif_composition[motif] += 1.0
        data.append(motif_composition)
    return data

def gappy_pair_data(sequences, maximum_distance, **args) :
    """generate a dataset object that contains all pairs of letters in a sequence that are
    within a certain distance of each other.

    :Parameters:
      - `sequences` - a list of sequences from which to construct the gappy pair
        representation or a Fasta file that contains the sequences
      - `maximum_distance` - the maximum distance between pairs of  length of the
        substrings to consider

    :Keywords:
      - `normalize` - whether to normalize the dataset [default: True]
      - `prefix` - a string to be added to the name of each feature 
        (useful when combining spectrum features from several sources)
      - `skip` - a list of characters that should be skipped in computing
        the features
    """
    prefix = ''
    if 'prefix' in args :
        prefix = args['prefix']
    normalize = True
    if 'normalize' in args :
        normalize = args['normalize']
    skip = []
    if 'skip' in args :
        skip = args['skip']
    if type(sequences) == type('') :
        sequences = fasta_read(sequences)

    data = SparseDataSet(generate_gappy_pairs(sequences, maximum_distance, prefix, skip))
    if normalize :
        data.normalize(2)
    return data

def generate_positional_kmers(sequences, k, prefix = '', skip = [], weight = 1.0,
                              shift_flag = False, shift_start = 1, shift_end = 1) :
    data = []
    for s in sequences :
        kmers = {}
        for i in range(len(s) - k + 1) :
            kmer = s[i:i+k]
            skip_flag = False
            if len(skip) > 0 :
                for c in kmer :
                    if c in skip : skip_flag = True
            if skip_flag :
                continue
            if len(prefix) > 0 : 
                kmer = prefix + kmer
            key = kmer + str(i)
            if key not in kmers :
                kmers[key] = 0.0
            kmers[key] += weight
            if (shift_flag and i >= shift_start and i < shift_end) :
                for shift in [-1, 1] :
                    key = kmer + str(i + shift)
                    if key not in kmers :
                        kmers[key] = 0.0
                    kmers[key] += weight
        data.append(kmers)
    return data

def positional_kmer_data(sequences, k1, k2=None, **args) :
    """generate a dataset object that represents kmers that occur in specific
    positions.  When using weighting, this is essentially the 'weighted degree'
    kernel of Sonenburg et al.

    :Parameters:
      - `sequences` - the name of a fasta file that contains the sequences or
         a list of sequences
      - `k1` - the smallest length kmer to consider
      - `k2` - if k2 is provided then strings whose length is between k1
        and k2 are used in constructing the kernel

    :Keywords:
      - `normalize` - whether to normalize the dataset [default: True]
      - `prefix` - a string to be added to the name of each feature 
        (useful when combining spectrum features from several sources)
      - `skip` - a list of characters that should be skipped in computing
        the spectrum [default: []].  Whenever a character in this list is
        encountered, the substring is not included in the spectrum
      - `weighted` - whether to use the weighting of sonenborg et al
        [default:  equal weights]
      - `shift` - whether to consider a shift [default: False]
      - `shift_start` - the position in the seq to start shifting [default: 0]
      - `shift_end` - the position in the sequence to stop shifting
        [default: end of sequence]
    """
    prefix = ''
    if 'prefix' in args :
        prefix = args['prefix']
    normalize = True
    if 'normalize' in args :
        normalize = args['normalize']
    if k2 is None : 
        k2 = k1 + 1
    skip = []
    if 'skip' in args :
        skip = args['skip']
    shift = False
    if 'shift' in args :
        shift = args['shift']
    shift_start = 0
    if 'shift_start' in args :
        shift_start = args['shift_start']
        
    if type(sequences) == type('') :
        sequences = fasta_read(sequences)

    if 'shift_end' in args :
        shift_end = args['shift_end']
    else :
        shift_end = len(sequences[0]) - 1

    weighted = False
    if weighted in args :
        weighted = args['weighted']
    if weighted :
        weights = [1.0 for i in range(k1, k2 + 1)]
    else :
        weights = [1.0 for i in range(k1, k2 + 1)]
    data = SparseDataSet(generate_positional_kmers(
        sequences, k1, prefix, skip, weights[0], shift, shift_start, shift_end))
    if normalize :
        data.normalize(2)
    for k in range(k1 + 1, k2 + 1) :
        data2 = SparseDataSet(generate_positional_kmers(
            sequences, k, prefix, skip, weights[k - k1], shift, shift_start, shift_end))
        if normalize :
            data2.normalize(2)
        data.addFeatures(data2)

    return data

