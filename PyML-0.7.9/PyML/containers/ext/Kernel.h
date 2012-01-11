# ifndef KERNEL_H
# define KERNEL_H

# include <vector>
# include <string>
# include <iostream>
# include <fstream>

# include "DataSet.h"

using namespace std;

class DataSet;

enum { NONE, COSINE, TANIMOTO, DICES };

class Kernel {
 public:

  Kernel();
  Kernel(const Kernel& other);
  virtual ~Kernel();
  
  virtual Kernel* duplicate() = 0;   //"virtual" copy constructor

  virtual Kernel* castToBase() = 0;

  virtual double eval(DataSet *datai, int i, int j, DataSet *dataj) = 0;

  int normalization;
  double normalize(double value, DataSet *datai, int i, int j, DataSet *dataj);

};

// Linear kernel:  k(x,y) = x.y

class Linear : public Kernel
{
 public:

  Linear();
  Linear(const Linear &other);
  ~Linear();

  virtual Linear* duplicate()
    {return new Linear(*this); }

  Kernel* castToBase() { return dynamic_cast<Kernel *>(this); }

  double eval(DataSet *datai, int i, int j, DataSet *dataj);

};

// Cosine kernel:  k(x,y) = x.y / \sqrt(x.x + y.y)

class Cosine : public Kernel
{
 public:

  Cosine();
  Cosine(const Cosine &other);
  ~Cosine();

  virtual Cosine* duplicate()
    {return new Cosine(*this); }

  Kernel* castToBase() { return dynamic_cast<Kernel *>(this); }

  double eval(DataSet *datai, int i, int j, DataSet *dataj);

};


// Polynomial $k(x,y) = (x.y + additiveConst)^degree$.

class Polynomial : public Kernel
{
 public:
  
  int degree;
  double additiveConst;
    
  Polynomial(const int degree_ = 2, const double additiveConst_ = 1);
  Polynomial(const Polynomial& other);
  ~Polynomial();

  virtual Polynomial* duplicate()
    {return new Polynomial(*this); }

  Kernel* castToBase() { return dynamic_cast<Kernel *>(this); }

  double eval(DataSet *datai, int i, int j, DataSet *dataj);

  void setDegree(const int degree_);
  void setAdditiveConst(const double additiveConst_);
};

// Gaussian $k(x,y) = exp(-gamma * ||x-y||^2)$

class Gaussian : public Kernel
{
 public:

  double gamma;

  Gaussian(const double gamma_ = 1.0);
  Gaussian(const Gaussian& other);
  ~Gaussian();

  virtual Gaussian* duplicate()
    {return new Gaussian(*this); }

  Kernel* castToBase() { return dynamic_cast<Kernel *>(this); }

  double eval(DataSet *datai, int i, int j, DataSet *dataj);

  void setGamma(const double gamma_);

};


class KernelMatrix {
 public:
  
  std::vector<std::vector<float> > matrix;
  int refCount;
  int size() { return matrix.size(); }

  void addRow(const std::vector<float>& row) 
    { matrix.push_back(row); }
  
  float getEntry(int i, int j) { return matrix[i][j]; }

  void center();

  void show();

  KernelMatrix();
  ~KernelMatrix();

};

void kernel2file(DataSet *data, char *fileName);


# endif
