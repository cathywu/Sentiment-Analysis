#ifndef SETDATA_H
#define SETDATA_H

# include <iostream>
# include <algorithm>
# include <vector>
# include <string>

# include "Kernel.h"
# include "DataSet.h"

using namespace std;

class SetData : public DataSet { 
 public:
    vector< vector<int> > sets;

    DataSet *data;

    int size() { return sets.size(); }

    void add(const std::vector<int>& set) { sets.push_back(set); }
  
    double dotProduct(int i, int j) ;
    double dotProduct(int i, int j, DataSet *other) ;

    void show();

    DataSet* castToBase() { return dynamic_cast<SetData *>(this); }

    SetData* duplicate(const std::vector<int> &patterns) 
    { return new SetData(*this, patterns); }

    SetData(int size, DataSet *data);

    SetData(const SetData &other, const std::vector<int> &patterns);

    ~SetData();
  
};


# endif
