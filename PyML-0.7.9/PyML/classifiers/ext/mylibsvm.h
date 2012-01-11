#ifndef MYLIBSVM_H
#define MYLIBSVM_H

#include "libsvm.h"
#include "../../containers/ext/SparseFeatureVector.h"
#include "../../containers/ext/FeatureVector.h"
#include "../../containers/ext/SparseDataSet.h"
#include "../../containers/ext/VectorDataSet.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <float.h>
#include <stdarg.h>
#include <math.h>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <vector>

void libsvm_construct(SparseDataSet &data, struct svm_problem &prob);
void libsvm_destroy(struct svm_problem &prob);
void libsvm_construct(VectorDataSet &data, svm_problem &prob);


#endif
