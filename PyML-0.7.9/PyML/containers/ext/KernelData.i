%module ckerneldata
%{
#include "DataSet.h"
#include "KernelData.h"
%}

class Kernel;

class KernelMatrix;

%include "std_vector.i"

namespace std
{
   %template(IntVector) vector<int>;
   %template(DoubleVector) vector<double>;   
   %template(FloatVector) vector<float>;
}

using namespace std;

class DataSet {
 public:

  virtual int size() = 0;
  // labels:
  std::vector<double> Y;     // numeric label
  std::vector<double> norms;

  void setY(int i, double c);
  void computeNorms();

  Kernel *kernel;

  void setKernel(Kernel *kernel_);
  void attachKernel(const DataSet& other);
  void attachKernel(Kernel* kernel_);
  std::vector<double> getKernelMatrixAsVector();

  virtual double dotProduct(int i, int j, DataSet *other = 0) = 0;

  virtual DataSet* duplicate(const std::vector<int> &patterns) = 0;
  virtual DataSet* castToBase() = 0;
  virtual void show() = 0;

  virtual DataSet(const int n);
  virtual DataSet(const DataSet &other, const std::vector<int> &patterns);
  DataSet();
  virtual ~DataSet();

};

class KernelData : public DataSet { 
 public:

  KernelMatrix *kernelMatrix;

  // patternPtr tells which patterns from the kernel matrix make up this dataset
  std::vector<int> patternPtr;

  // number of examples:
  int size() { return patternPtr.size(); }

  virtual double dotProduct(int i, int j, DataSet *other = 0) ;

  void center();
	
  void show();
  
  void addRow(std::vector<float> &values) { kernelMatrix->addRow(values); }

  DataSet* castToBase() { return dynamic_cast<DataSet *>(this); }

  virtual KernelData* duplicate(const std::vector<int> &patterns) 
    { return new KernelData(*this, patterns); }

  KernelData();
  KernelData(KernelMatrix *kernelMatrix_);

  KernelData(const KernelData &other, const std::vector<int> &patterns);

  ~KernelData();

};



