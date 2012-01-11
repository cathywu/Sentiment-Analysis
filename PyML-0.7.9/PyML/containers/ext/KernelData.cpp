
# include "KernelData.h"


KernelData::KernelData() :
  kernelMatrix(0)
{
}

KernelData::KernelData(KernelMatrix *kernelMatrix_) : 
  DataSet(kernelMatrix_->size()),
  kernelMatrix(kernelMatrix_),
  patternPtr(kernelMatrix->size())
{
  ++kernelMatrix->refCount;
  for (int i = 0; i < kernelMatrix->size(); ++i) {
    patternPtr[i] = i;
  }
}


KernelData::KernelData(const KernelData &other, const std::vector<int> &patterns) :
  DataSet(other, patterns),
  kernelMatrix(other.kernelMatrix),
  patternPtr(patterns.size())
{
  ++kernelMatrix->refCount;
  for (unsigned int i = 0; i < patterns.size(); ++i) {
    patternPtr[i] = other.patternPtr[patterns[i]];
  }
}

KernelData::~KernelData() 
{
  //cout << "in KernelData::~KernelData" << endl;
  if (--kernelMatrix->refCount == 0) {
    delete kernelMatrix;
  }
}

double KernelData::dotProduct(int i, int j)
{
  KernelData *other;
    other = this;

  return dotProduct(i, j, other);

}  


double KernelData::dotProduct(int i, int j, DataSet *other_)
{
  KernelData *other;
  if (other_ == 0) 
    other = this;
  else 
    other = dynamic_cast<KernelData *>(other_);

  return kernelMatrix->getEntry(this->patternPtr[i], other->patternPtr[j]);

}  

void KernelData::show() 
{
  kernelMatrix->show();
}

