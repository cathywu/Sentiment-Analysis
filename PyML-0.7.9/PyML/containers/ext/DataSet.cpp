
# include "DataSet.h"

DataSet::DataSet() : kernel(0) { }

DataSet::DataSet(const DataSet &other, const std::vector<int> &patterns) : 
  kernel(other.kernel->duplicate()),
  Y(patterns.size(), 0),
  norms(patterns.size(), 0)
{ 
  //cout << "copying norms" << endl;
  //cout << "norms size: " << norms.size() << endl;
  for (int i = 0; i < patterns.size(); i++) {
	norms[i] = other.norms[patterns[i]];
  }
  //cout << "done DataSet:DataSet" << endl;
}

DataSet::DataSet(const int numPatterns) :  
  kernel(0),
  Y(numPatterns, 0),
  norms(numPatterns, 0)
{ }

DataSet::~DataSet() 
{
  //cout<< "in DataSet::~DataSet" << endl;
  delete kernel;
}

void DataSet::setY(int i, double c) 
{
  Y[i] = c;
}

void DataSet::computeNorms()
{
  cout << "computing norms" << endl;
  cout << "size: " << norms.size() << endl;
  for (int i = 0; i < size(); ++i) {
	norms[i] = dotProduct(i, i);
  }
}

void DataSet::attachKernel(Kernel *kernel_) 
{
  if (kernel != 0) {
    delete kernel;
  }
  kernel = kernel_->duplicate();
}

void DataSet::attachKernel(const DataSet& other) 
{
  if (kernel != 0) {
    delete kernel;
  }
  kernel = other.kernel->duplicate();
}


std::vector<double> DataSet::getKernelMatrixAsVector()
{
  std::vector<double> kmat(this->size() * this->size());

  for (int i = 0; i < size(); ++i) {
    for (int j = i; j < size(); ++j) {
      kmat[i * size() + j] = this->kernel->eval(this, i, j, this);
      kmat[j * size() + i] = kmat[i * size() + j];
    }
  }
  return kmat;
}
