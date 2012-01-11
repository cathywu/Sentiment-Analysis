# ifndef KERNELDATA_H
# define KERNELDATA_H

# include <iostream>
# include <algorithm>
# include <vector>
# include <string>

# include "Kernel.h"
# include "DataSet.h"
//# include "RefCount.h"

using namespace std;

class KernelData : public DataSet { 
 public:

  KernelMatrix *kernelMatrix;

  // patternPtr tells which patterns from the kernel matrix make up this dataset
  std::vector<int> patternPtr;

  // number of examples:
  int size() { return patternPtr.size(); }

  double dotProduct(int i, int j);
  double dotProduct(int i, int j, DataSet *other) ;

  void center() { kernelMatrix->center(); }

  void show();
  //void save(char *fileName);
  
  void addRow(std::vector<float> &values) { kernelMatrix->addRow(values); }

  DataSet* castToBase() { return dynamic_cast<DataSet *>(this); }

  virtual KernelData* duplicate(const std::vector<int> &patterns) 
    { return new KernelData(*this, patterns); }

  KernelData();
  KernelData(KernelMatrix *kernelMatrix_);

  KernelData(const KernelData &other, const std::vector<int> &patterns);

  ~KernelData();

};


# endif
