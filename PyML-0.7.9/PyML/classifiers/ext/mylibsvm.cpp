# include "mylibsvm.h"


void libsvm_destroy(struct svm_problem &prob) {

  for (int i = 0; i < prob.l; i++) {
    delete [] prob.x[i];
  }
  delete [] prob.x;
  delete [] prob.y;
}


