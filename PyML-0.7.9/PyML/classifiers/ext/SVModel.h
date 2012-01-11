
# ifndef SVMODEL_H
# define SVMODEL_H

# include "../../containers/ext/DataSet.h"
# include "../../containers/ext/SparseDataSet.h"
# include "../../containers/ext/SparseFeatureVector.h"
# include "../../containers/ext/VectorDataSet.h"
# include "../../containers/ext/FeatureVector.h"
# include "../../containers/ext/Kernel.h"

# include <vector>

using namespace std;

class SVModel {
 public:
  DataSet *svdata;
  vector<double> alpha;
  double b;
  int numSV;

  double decisionFunc (DataSet *data, int i);

  SVModel(DataSet &data, const std::vector<int>& svID, 
	  const std::vector<double>& alpha_, double b_);

  ~SVModel();

};

class LinearSVModel {
 public:

  LinearSVModel(VectorDataSet& data, const std::vector<int>& svID, 
		const std::vector<double>& alpha_, double b_);

  ~LinearSVModel();

  //ASA:  why both w and wVec?

  FeatureVector w;
  std::vector<double> wVec;
  double b;

  void getW(std::vector<double> &values);
  std::vector<double> getWvec();

  double decisionFunc (FeatureVector &x);
  double decisionFunc (VectorDataSet &data, int i);

};


class LinearSparseSVModel {
 public:

  LinearSparseSVModel(SparseDataSet& data, const std::vector<int>& svID, 
		      const std::vector<double>& alpha_, double b_);
  LinearSparseSVModel(SparseDataSet& data, const std::vector<double>& w_, double b_);

  ~LinearSparseSVModel();

  SparseFeatureVector w;
  std::vector<double> wVec;
  double b;

  void getW(std::vector<double> &values, std::vector<long> &ids);
  std::vector<double> getWvec();

  double decisionFunc (SparseFeatureVector &x);
  double decisionFunc (SparseDataSet &data, int i);

};

# endif
