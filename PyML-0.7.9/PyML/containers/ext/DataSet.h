#ifndef DATASET_H
#define DATASET_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <float.h>
#include <stdarg.h>
#include <math.h>
# include <cmath>

# include <iostream>
# include <algorithm>
# include <vector>
# include <string>

# include "Kernel.h"

using namespace std;

class Kernel;

class DataSet {
 public:
  std::vector<double> Y;
  std::vector<double> norms;
  Kernel *kernel;

  DataSet(const int n);
  DataSet(const DataSet &other, const std::vector<int> &patterns);
  DataSet();

  void setY(int i, double c);
  void computeNorms();

  void setKernel(Kernel *kernel_) { kernel = kernel_; }
  void attachKernel(const DataSet& other);
  void attachKernel(Kernel* kernel_);

  std::vector<double> getKernelMatrixAsVector();

  virtual int size() = 0;  
  virtual double dotProduct(int i, int j) = 0;
  virtual double dotProduct(int i, int j, DataSet* other) = 0;

  virtual DataSet* duplicate(const std::vector<int> &patterns) = 0;

  virtual DataSet* castToBase() = 0;
  virtual void show() = 0;

  virtual ~DataSet();

};


#endif
