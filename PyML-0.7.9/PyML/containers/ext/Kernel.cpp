
# include "Kernel.h"

Kernel::Kernel() : normalization(NONE)
{
  //cout << "in Kernel::kernel()" << endl;
}

Kernel::Kernel(const Kernel& other) : 
  normalization(other.normalization)
{ 
  //cout << "in Kernel::kernel(other) " << other.normalization << endl; 
}

Kernel::~Kernel()
{ }

double Kernel::normalize(double kij, DataSet *datai, int i, int j, DataSet *dataj)
{
  int temp = normalization;
  normalization = NONE;
  double kii = eval(datai, i, i, datai);
  double kjj = eval(dataj, j, j, dataj);
  normalization = temp;

  if (kii == 0 || kjj == 0) {
	return 0;
  }
  if (normalization == COSINE) {
    return kij / sqrt(kii * kjj);
  }
  if (normalization == TANIMOTO) {
    return kij / (kii + kjj - kij);
  }
  if (normalization == DICES) {
    return 2 * kij / (kii + kjj);
  }
    
}

Linear::Linear() 
{ }

Linear::Linear(const Linear& other) : 
  Kernel(other)
{ }

double Linear::eval(DataSet *datai, int i, int j, DataSet *dataj)
{
  if (normalization == NONE) {
	return datai->dotProduct(i, j, dataj);
  }
  return normalize(datai->dotProduct(i, j, dataj), datai, i, j, dataj);
}

Linear::~Linear()
{ }


Cosine::Cosine() 
{ }

Cosine::Cosine(const Cosine& other) :
  Kernel(other)
{ }

double Cosine::eval(DataSet *datai, int i, int j, DataSet *dataj)
{
  if (dataj == 0) dataj = datai;
  double kii = datai->norms[i];
  double kjj = dataj->norms[j];
  if (kii == 0 || kjj == 0) {
    return 0;
  }
  return datai->dotProduct(i, j, dataj) / sqrt(kii * kjj);
}

Cosine::~Cosine()
{ }

Polynomial::Polynomial(const int degree_, const double additiveConst_) : 
  degree(degree_), additiveConst(additiveConst_)
{ }

Polynomial::Polynomial(const Polynomial& other) : 
  Kernel(other),
  degree(other.degree), 
  additiveConst(other.additiveConst)
{ }
  
Polynomial::~Polynomial()
{ }

double Polynomial::eval(DataSet *datai, int i, int j, DataSet* dataj)
{
  double arg;
  if ((i == j) && (datai == dataj)) {
    arg = datai->norms[i] + additiveConst;
  }
  else {
    arg = datai->dotProduct(i, j, dataj) + additiveConst; 
  }
  double res = arg;
  for (int k = 1; k < degree; k++){
    res = arg*res;
  }
  if (normalization == NONE) {
    return res;
  }
  return normalize(res, datai, i, j, dataj);

}

void Polynomial::setDegree(const int degree_) 
{
  degree = degree_;
}

void Polynomial::setAdditiveConst(const double additiveConst_) 
{
  additiveConst = additiveConst_;
}


Gaussian::Gaussian(const double gamma_) : gamma(gamma_)
{ 
}

Gaussian::Gaussian(const Gaussian& other) : 
  Kernel(other),
  gamma(other.gamma)
{
}

Gaussian::~Gaussian() 
{ }

double Gaussian::eval(DataSet *datai, int i, int j, DataSet* dataj)
{
  return (exp(gamma * (2 * datai->dotProduct(i, j, dataj) - 
					   datai->norms[i] - dataj->norms[j] )));
  //datai->dotProduct(i, i, datai) - 
  //dataj->dotProduct(j, j, dataj))));
}

void Gaussian::setGamma(const double gamma_) 
{
  gamma = gamma_;
}


KernelMatrix::KernelMatrix() : refCount(0)
{ }

KernelMatrix::~KernelMatrix()
{ }

void KernelMatrix::show()
{
  cout << "in KernelMatrix::show() " << endl;
  for (unsigned int i = 0; i < matrix.size(); ++i) {
    for (unsigned int j = 0; j < matrix[i].size(); ++j) {
      cout << matrix[i][j] << " ";
    }
    cout << endl;
  }
}

void KernelMatrix::center()
//  * expressed in terms of the original matrix K (of dimensionality M)
//  * as follows:
//  *
//  * \~K_{ij} = [\phi(x_i) - 1/M \sum_{m=1}^M \phi(x_m)]
//  *             . [\phi(x_j) - 1/M \sum_{n=1}^M \phi(x_n)]
//  *
//  *          = [\phi(x_i) \phi(x_j)] - [1/M \phi(x_i) \sum_{m=1}^M \phi(x_m)]
//  *             - [1/M \phi(x_j) \sum_{n=1}^M \phi(x_n)]
//  *             + [1/M^2 \sum_{m=1}^M \sum{n=1}^M \phi(x_m) \phi(x_n)]
//  *
//  *          = [K_{ij}] - [1/M \sum_{m=1}^M K_{mj}] - [1/M \sum_{n=1}^M K_{in}]
//  *             + [1/M^2 \sum_{m=1}^M \sum{n=1}^M K_{mn}]
//  *
{
  cout << "centering..." << endl;
  vector<float> rowAvg(size(), 0);
  float avgEntry = 0;

  for (int i = 0; i < size(); ++i) {
    for (int j = 0; j < size(); ++j) {
      rowAvg[i] += getEntry(i, j);
    }
    rowAvg[i] /= size();
    avgEntry += rowAvg[i];
  }
  avgEntry /= size();

  for (int i = 0; i < size(); ++i) {
    for (int j = 0; j < size(); ++j) {
      matrix[i][j] = matrix[i][j] - rowAvg[i] - rowAvg[j] + avgEntry;
    }
  }
}

void kernel2file(DataSet *data, char *fileName)
{

  ofstream out(fileName);
  for (int i = 0; i < data->size(); ++i) {
    for (int j = 0; j < data->size(); ++j) {
      out << "\t" << data->kernel->eval(data, i, j, data);
    }
    out << endl;
  }

}

