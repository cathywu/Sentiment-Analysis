%module mylibsvm
%{
#include "mylibsvm.h"
%}

void libsvm_destroy(struct svm_problem &prob);


