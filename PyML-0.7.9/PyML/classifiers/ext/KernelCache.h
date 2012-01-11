# ifndef KERNELCACHE_H
# define KERNELCACHE_H

# include "../../containers/ext/DataSet.h"
# include "../../containers/ext/Kernel.h"

# include <list>
# include <set>
# include <ext/hash_set>
# include <vector>
# include <algorithm>
# include <functional>
#include <string>
#include <iostream>
#include <fstream>
# include <cmath>

#define INF HUGE_VAL

using namespace std;

class KernelCache {
 public:
  DataSet *data;
  vector<vector<float> > rows;   // the rows of the kernel matrix
  list<int> lru;             // a list whose head is the most recently accessed row
  vector<list<int>::iterator > lruPtr; 
  // a vector that tells the position of each row in the priority list
  vector<int> rowPtr;     // which row in the cache a row in the matrix is stored in

  int cacheMemorySize;    //size of cache in MB
  int length;             //number of patterns in dataset
  int numCacheable;       //number of rows that can fit in the cache
  int numCached;

  // number of patterns in cache
  int size() { return rows.size(); }

  vector<bool> _cached;
  bool isCached(int i) { return _cached[i]; }
  void setCached(int i, bool state) { _cached[i] = state; }

  vector<float>& getRow(int i);
  
  KernelCache(DataSet *data_, int cacheMemorySize_);
  ~KernelCache();

};

# endif
