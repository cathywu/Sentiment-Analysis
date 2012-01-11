#ifndef AGGREGATE_H
#define AGGREGATE_H

# include <iostream>
# include <algorithm>
# include <vector>
# include <string>

//# include "Vector.h"
# include "Kernel.h"
# include "DataSet.h"

using namespace std;

class Kernel;

class Aggregate : public DataSet { 
 public:

  vector<DataSet *> datas;
  vector<double> weights;
  bool ownData;

  int size() { return Y.size(); }

  double dotProduct(int i, int j);
  double dotProduct(int i, int j, DataSet *other) ;

  void show();

  DataSet* castToBase() { return dynamic_cast<DataSet *>(this); }

  virtual Aggregate* duplicate(const std::vector<int> &patterns) 
    { return new Aggregate(*this, patterns); }

  void addDataSet(DataSet *data) { datas.push_back(data); }
  void addDataSet(DataSet *data, double weight) 
    { datas.push_back(data); weights.push_back(weight); }

  Aggregate() {}
  Aggregate(int n);
  Aggregate(int n, const std::vector<double> &weights);
  //Aggregate(vector<DataSet *> &datas_);
  Aggregate(const Aggregate &other, const std::vector<int> &patterns);

  ~Aggregate();

};

#endif
