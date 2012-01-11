
from PyML.utils.ext import carrayWrap
import numpy

def arrayFromDoubleArray(doubleArray, length) :

    array = numpy.zeros(length, numpy.float_)
    for i in range(length) :
        array[i] = doubleArray[i]
    return array


def doubleArray(a) :

    array = carrayWrap.doubleArray(len(a))
    for i in range(len(a)) :
        array[i] = a[i]

    return array

def intArray(a) :

    array = carrayWrap.intArray(len(a))
    for i in range(len(a)) :
        array[i] = a[i]

    return array

def doubleVector(v = None) :

    if v is None : return carrayWrap.DoubleVector()
    vector = carrayWrap.DoubleVector(len(v))
    for i in range(len(v)) :
        vector[i] = v[i]

    return vector

def floatVector(v = None) :

    if v is None : return carrayWrap.FloatVector()
    vector = carrayWrap.FloatVector(len(v))
    for i in range(len(v)) :
        vector[i] = v[i]

    return vector

def intVector(v = None) :

    if v is None : return carrayWrap.IntVector()
    vector = carrayWrap.IntVector(len(v))
    for i in range(len(v)) :
        vector[i] = int(v[i])
    return vector


def longVector(v = None) :

    if v is None : return carrayWrap.LongVector()
    vector = carrayWrap.LongVector(len(v))
    for i in range(len(v)) :
        vector[i] = v[i]

    return vector

def datasetVector(v) :

    if v is None : return carrayWrap.DataSetPtrVector()
    vector = carrayWrap.DataSetPtrVector(len(v))
    for i in range(len(v)) :
        vector[i] = v[i]

    return vector

def intVector2list(v) :

    #vPtr = carrayWrap.IntVectorPtr(v)
    vPtr = carrayWrap.IntVector(v)
    return [vPtr[i] for i in range(len(vPtr))]

def doubleVector2list(v) :

    #vPtr = carrayWrap.DoubleVectorPtr(v)
    vPtr = carrayWrap.DoubleVector(v)
    return [vPtr[i] for i in range(len(vPtr))]

def dict2vectors(a) :

    keys = carrayWrap.LongVector(len(a))
    #keys = carrayWrap.IntVector(len(a))
    values = carrayWrap.DoubleVector(len(a))
    keyList = a.keys()
    keyList.sort()
    for i in range(len(keyList)) :
        keys[i] = keyList[i]
        values[i] = a[keys[i]]

    return keys,values
        

doubleArray_frompointer = carrayWrap.doubleArray_frompointer
