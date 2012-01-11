# include "KernelCache.h"

KernelCache::KernelCache(DataSet *data_, int cacheMemorySize_) : 
  data(data_), 
  length(data->size()), 
  cacheMemorySize(cacheMemorySize_),
  numCached(0),
  _cached(data->size(), false),
  rows(data->size()),
  rowPtr(data->size(), 0),
  lruPtr(data->size())
{
  numCacheable = int(float(cacheMemorySize) * 1024.0 * 1024.0 / 
		     float(sizeof(float) * length));
  cout << "numCacheable " << numCacheable << endl;
}

KernelCache::~KernelCache()
{
  //delete data;
}

vector<float>& KernelCache::getRow(int i)
{
  //cout << "gettingRow " << i << endl;
  if (isCached(i)) {
    //cout << "cached" << endl;
    // remove pattern i from its current position in the list
    lru.erase(lruPtr[i]);
  }
  else {
    //cout << "numCached: " << numCached << endl;
    if (numCached >= numCacheable) {  // need to erase something
      //cout << "erasing..." << endl;
      int elementToErase = lru.back();
      setCached(elementToErase, false);
      rowPtr[i] = rowPtr[elementToErase];
      lru.pop_back();
    }
    else {
      // create the new row:
      //cout << "creating row..." << endl;
      rowPtr[i] = numCached;
      rows[numCached] = vector<float>(length);
      ++numCached;
    }
    setCached(i, true);
    for (int j = 0; j < length; j++) {
      rows[rowPtr[i]][j] = data->kernel->eval(data, i, j, data);
    }
  }
  lru.insert(lru.begin(), i);
  lruPtr[i] = lru.begin();
  //cout << "finished get row" << endl;

  vector<float> &retval = rows[rowPtr[i]];
  return retval;
}

