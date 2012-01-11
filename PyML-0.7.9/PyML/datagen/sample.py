
import random
from PyML.utils import misc

"""
a collection of functions for sampling from a dataset
"""

def shuffle(x) :
    """
    shuffle a list
    """

    shuffled = x[:]
    random.shuffle(shuffled)

    return shuffled

def sample(data, size, **args) :
    """
    sample from a dataset without replacement

    :Parameters:
      - `data` - a dataset object
      - `size` - can be one of the following:
        An integer - in this case the given number of patterns are chosen.
        A list - size[i] specifies how many examples to sample from
        class i (data.labels.classLabels will tell how they are indexed).
        A dictionary whose keys are the class names e.g. {'+1': 100, '-1':100}.
        If an entry in the list or dictionary is 'all' then all members of
        the corresponding class are sampled.

    :Keywords:
      - `stratified` - whether to perform stratified sampling [default: True].
        This applies only when a global 'size' parameter is provided
      - `seed` - random number generator seed      
    """

    stratified = True
    if 'stratified' in args :
	stratified = args['stratified']
    if 'seed' in args :
        seed = args['seed']
        rand = random.Random(seed)
    else :
        rand = random.Random()

    patterns = []
    if type(size) == type(1) :
        if stratified :
            fraction = float(size) / float(len(data))
            patterns = []
            for i in range(data.labels.numClasses) :
                if i < data.labels.numClasses - 1 :
                    numToSample = int(fraction * data.labels.classSize[i])
                else :
                    numToSample = size - len(patterns)
                I = data.labels.classes[i][:]
                rand.shuffle(I)
                patterns.extend(I[:numToSample])
        else :
            I = range(len(data))
            rand.shuffle(I)
            patterns = I[:size]
    elif type(size) == type([]) :
        for i in range(len(size)) :
            if size[i] == 'all' :
                patterns.extend(data.labels.classes[i][:])
            else :
                I = data.labels.classes[i][:]
                rand.shuffle(I)
                patterns.extend(I[:size[i]])
    elif type(size) == type({}) :
        for classLabel in size :
            if size[classLabel] == 'all' :
                patterns.extend(data.labels.classes[data.labels.classDict[
                    classLabel]][:])
            else :
                I = data.labels.classes[data.labels.classDict[classLabel]][:]
                rand.shuffle(I)
                patterns.extend(I[:size[classLabel]])
            
    return data.__class__(data, patterns = patterns)


def split(data, fraction, **args) :
    """
    split a dataset into training and test sets.
    randomly splits a dataset into two datasets whose sizes are determined
    by the 'fraction' parameter (the first dataset will contain that fraction
    of the examples).

    for example:
    train, test = split(data, 0.7)
    will split the data -- 70% for training and 30% for test

    :Parameters:
      - `data` - a dataset object
      - `fraction` - the fraction of the examples to put in the first split

    :Keywords:
      - `stratified` - whether to perform stratified splitting, i.e. whether to 
        keep the class ratio in the two datasets [default: True]
      - `seed` - random number generator seed
      - `indicesOnly` - if this flag is set, the indices of the two splits are
        returned instead of the datasets [default: False]
    """

    if 'seed' in args :
        seed = args['seed']
        rand = random.Random(seed)
    else :
        rand = random.Random()

    indicesOnly = False
    if 'indicesOnly' in args :
        indicesOnly = args['indicesOnly']

    if data.__class__.__name__ == 'Labels' :
        labels = data
    else :
        labels = data.labels

    sampleSize = int(len(data) * fraction)

    stratified = True
    if 'stratified' in args :
        stratified = args['stratified']

    if stratified :
        patterns = []
	for i in range(labels.numClasses) :
	    if i < labels.numClasses - 1 :
                numToSample = int(fraction * labels.classSize[i])
	    else :
                numToSample = sampleSize - len(patterns)
            I = labels.classes[i][:]
            rand.shuffle(I)
            patterns.extend(I[:numToSample])
    else :
        I = range(len(data))
        rand.shuffle(I)
        patterns = I[:sampleSize]
    patterns.sort()
        
    if not indicesOnly :
        return (data.__class__(data, patterns = patterns), 
                data.__class__(data, patterns = misc.setminus(range(len(data)), patterns) ) )
    else :
        return patterns, misc.setminus(range(len(data)), patterns)


def bootstrap(data, **args) :
    """
    return a bootstrap sample from a dataset

    :Parameters:
      - `data` - a dataset object

    :Keywords:
      - `stratified` - whether to perform stratified bootstrapping, i.e. whether to 
        keep the class ratio
      - `seed` - random number generator seed
    """

    if 'seed' in args :
        seed = args['seed']
        rand = random.Random(seed)
    else :
        rand = random.Random()
    stratified = True
    if 'stratified' in args :
        stratified = args['stratified']
    if not data.labels.isLabeled() :
        stratified = False

    if not stratified :
        patterns = [rand.randint(0, len(data) - 1) for i in range(len(data))]
    else :
        patterns = []
        for c in range(len(data.labels.classLabels)) :
            classSize = len(data.labels.classes[c])
            patterns.extend([data.labels.classes[c][rand.randint(0, classSize - 1)]
                             for i in range(classSize)])

    patterns.sort()
    return data.__class__(data, patterns = patterns)

