# ifndef Kmeans_H
# define Kmeans_H

# include <cstdlib>
# include <set>
# include <vector>
# include <algorithm>
# include <iostream>
# include <functional>

# include "../../containers/ext/DataSet.h"

typedef set<int> IntSet;

class Kmeans {
  public :

    Kmeans(int k_);
    int k;
    int max_iterations;

    vector<int> cluster_membership;
    //vector<hash_map<int, int>> clusters;
    vector< set<int> > clusters;

    float similarity_to_cluster(int example, DataSet* data, int cluster);

    void move(int example, int cluster);

    void initialize_clusters(DataSet* data);
    void show(void);

    std::vector<int> train(DataSet* data);
    std::vector<int> test(DataSet* testdata, std::vector<double>& decisionFunction);
			  


};

#endif
  
