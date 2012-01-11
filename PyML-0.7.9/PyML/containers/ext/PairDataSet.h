#ifndef PAIRDATASET_H
#define PAIRDATASET_H

# include <iostream>
# include <algorithm>
# include <vector>
# include <string>

# include "Kernel.h"
# include "DataSet.h"

using namespace std;

class PairDataSet : public DataSet { 
 public:

  // the pairs
  vector<int> first;
  vector<int> second;

  DataSet *data;

  // number of examples:
  int size() { return first.size(); }
  
	double dotProduct(int i, int j) ;
  double dotProduct(int i, int j, DataSet *other) ;

  void show();

  DataSet* castToBase() { return dynamic_cast<DataSet *>(this); }

  PairDataSet* duplicate(const std::vector<int> &patterns) 
    { return new PairDataSet(*this, patterns); }

  PairDataSet() {}
  PairDataSet(const std::vector<int>& first_, 
	      const std::vector<int>& second_, 
	      DataSet *data_);
  PairDataSet(const PairDataSet &other, const std::vector<int> &patterns);

  ~PairDataSet();

};


class PairDataSetSum : public PairDataSet { 
 public:
 
  double dotProduct(int i, int j);
  double dotProduct(int i, int j, DataSet *other);

  DataSet* castToBase() { return dynamic_cast<DataSet *>(this); }

  virtual PairDataSetSum* duplicate(const std::vector<int> &patterns) 
    { return new PairDataSetSum(*this, patterns); }

  PairDataSetSum() {}
  PairDataSetSum(const std::vector<int>& first_, 
		 const std::vector<int>& second_, 
		 DataSet *data_);
  PairDataSetSum(const PairDataSetSum &other, const std::vector<int> &patterns);

  ~PairDataSetSum();


};

class PairDataSetOrd : public PairDataSet { 
 public:
  double dotProduct(int i, int j) ;
  double dotProduct(int i, int j, DataSet *other);

  DataSet* castToBase() { return dynamic_cast<DataSet *>(this); }

  virtual PairDataSetOrd* duplicate(const std::vector<int> &patterns) 
    { return new PairDataSetOrd(*this, patterns); }

  PairDataSetOrd() {}
  PairDataSetOrd(const std::vector<int>& first_, 
		 const std::vector<int>& second_, 
		 DataSet *data_);
  PairDataSetOrd(const PairDataSetOrd &other, const std::vector<int> &patterns);

  ~PairDataSetOrd();


};

# endif
