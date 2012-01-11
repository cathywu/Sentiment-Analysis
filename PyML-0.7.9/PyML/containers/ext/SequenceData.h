#ifndef SequenceData_H
#define SequenceData_H

#include <string.h>
#include <iostream>
#include <vector>
#include <string>

#include "Kernel.h"
#include "DataSet.h"
//#include "StringKernel.h"

using namespace std;

//class StringKernel;

class SequenceData : public DataSet {
 private:

 public:
    vector<std::string> X;

    SequenceData();
    SequenceData(const int);
    SequenceData(const SequenceData &other, const std::vector<int> &patterns);
    ~SequenceData();

    std::string& operator [] (const int i) { return X[i]; }

    void addPattern(const std::string& sequence);
    std::string getSequence(const int i) { return X[i]; }

    int size() { return Y.size(); }
    DataSet* castToBase() { return dynamic_cast<DataSet *>(this); }
    void show();
  
    SequenceData* duplicate(const std::vector<int> &patterns){ 
  	return new SequenceData(*this, patterns); 
    }
    double dotProduct(int i, int j, DataSet* dataj);
    double dotProduct(int i, int j) { return dotProduct(i, j, this); }

    int mink;
    int maxk;
    int maxShift;        //the maximum allowed shift
    int noShiftStart;
    int noShiftEnd;
    int mismatches;
    std::vector<int> mismatchProfile;
    std::vector<double> shiftWeight;

    void setMismatchProfile(const std::vector<int>& profile) { mismatchProfile = profile; }
    void setShiftWeight(const std::vector<double>& weights) { shiftWeight = weights; }

    // compute the amount of shift at a particular position
    int shiftSize(const int pos, const int length);
    //    void addPattern(const std::string& sequence) { SequenceData::addPattern(sequence); }
  
};

#endif
