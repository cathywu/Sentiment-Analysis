
# ifndef KNN_H
# define KNN_H

# include <vector>
# include <algorithm>
# include <iostream>
# include <functional>

# include "../../containers/ext/DataSet.h"

class KNN {
  public :

    KNN(int k);
    DataSet* data;
    int k;
    int numClasses;
    
    void train(DataSet* _data);
    std::vector<double> test(DataSet& testdata);

    std::vector<double> classScores(DataSet& testdata, int p);
    int nearestNeighbor(DataSet& data, int pattern);
    std::vector<int> nearestNeighbors(DataSet& testdata, int p);

};

#endif
  
