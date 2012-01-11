# include "kmeans.h"

Kmeans::Kmeans(int k_) :  
  k(k_), max_iterations(500)
{

}

float Kmeans::similarity_to_cluster(int example, DataSet* data, int cluster)
{
  float similarity = 0;
  IntSet::iterator itr;
  int cluster_size = 0;
  for (itr = clusters[cluster].begin(); itr != clusters[cluster].end(); ++itr){
    if (example!=*itr) {
      similarity += data->kernel->eval(data, example, *itr, data);
      ++cluster_size;
    }
  }
  return similarity / float(cluster_size);
}

void Kmeans::move(int example, int new_cluster)
{

  int old_cluster = cluster_membership[example];
  clusters[old_cluster].erase(example);
  clusters[new_cluster].insert(example);
  cluster_membership[example] = new_cluster;

}

void Kmeans::show() 
{
  for (int i = 0; i < cluster_membership.size(); ++i) {
    cout << "example " << i << " belongs to cluster " << cluster_membership[i] << endl;
  }
  IntSet::iterator itr;
  for (int cluster=0; cluster < k; ++cluster) {
    cout << "example in cluster " << cluster << endl;
    for (itr = clusters[cluster].begin(); itr != clusters[cluster].end(); ++itr){
      cout << *itr << " ";
    }
    cout << endl;
  }
  

}

std::vector<int> Kmeans::train(DataSet* data)
{ 
  initialize_clusters(data);

  vector<float> similarities(k, 0);
  //show();
  for (int itr = 0; itr < max_iterations; ++itr) {
    cout << "iteration: " << itr << endl;
    bool moved = false;
    for (int i = 0; i < data->size(); ++i) {
      //cout << "i: " << i << endl;
      float maxSimilarity = -1e10;
      int newCluster = 0;
      for (int cluster = 0; cluster < k; ++cluster) {
	float s = similarity_to_cluster(i, data, cluster);
	//cout << "s: " << s << endl;
	if (s > maxSimilarity) {
	  newCluster = cluster;
	  maxSimilarity = s;
	}
      }
      if (newCluster != cluster_membership[i]) {
	move(i, newCluster);
	moved = true;
      }
    }
    //show();
    if (!moved) {
      break;
    }
  }
  return cluster_membership;

}

std::vector<int> Kmeans::test(DataSet* testdata, std::vector<double>& decisionFunction)
{

  std::vector<int> labels(testdata->size());
  decisionFunction.reserve(testdata->size());

  for (int i = 0; i < testdata->size(); i++) {
    vector<double> similarities(k, 0);
    for (int c = 0; c < k; c++) {
      similarities[c] = similarity_to_cluster(i, testdata, c);
    }
    double largestSimilarity = -1e10;
    for (int c = 0; c < k; c++) {
      if (similarities[c] > largestSimilarity) {
	largestSimilarity = similarities[c];
	labels[i] = c;
      }
      decisionFunction[i] = similarities[labels[i]];
    }

  }
  return labels;
}
			      
void Kmeans::initialize_clusters(DataSet* data)
{
  cluster_membership.clear();
  clusters.clear();
  for (int i = 0; i < k; ++i){
    set<int> s;
    clusters.push_back(s);
  }
  cout << "initializing" << endl;
  for (int i = 0; i < data->size(); ++i){
    int cluster_idx = rand() % k;
    cluster_membership.push_back( cluster_idx );
    clusters[cluster_idx].insert(i);
  }
}  

