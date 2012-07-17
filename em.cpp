#include <fstream>
ofstream fout("log.txt");

void save_init(matrix data){
  fout << 2 << endl;
  fout << data.num_rows() << data.num_cols() << endl;
  fout << data << endl;
}

void save(vector<ClusterStats> clusters,vector<int> assignments=NULL){
  int K=clusters.size();
  int N=assignments.size();
  int D=clusters[0].d;
  fout << (assignments==NULL?0:1) << endl;
  fout << N << D << K << endl;
  for(int k=0;k<K;k++)
    fout << clusters[k].mu() << endl;
    fout << clusters[k].sigma() << endl;
  }
  if(assignments != NULL)
    for(int n=0;n<N;n++)
      fout << assignments[n] << endl;
}

void em(matrix data,int K,int T=-1,bool debug=false){ // NxD
  int N=data.num_rows();
  int D=data.num_cols();

  //initialization
  vector<ClusterStats> clusters;
  for(int k=0;k<K;k++){
    new_cluster = ClusterStats(D,0.0,0.0,zeros(D,1),zeros(D,D));
    new_cluster.add(data[rand()%N][:]);
    clusters.push_back(new_cluster);
  }
  if(debug)
    save(clusters);

  for(int t=0;t!=T;t++){ //TODO deal with case when T == -1
    //update assignments
    vector<int> assignments;
    for(int n=0;n<N;n++){
      vector<double> logprobs;
      for(int k=0;k<K;k++)
        logprobs.push_back(clusters[i].logpdf_em(data[n][:]));
      assignments.push_back(sample(logprobs));
    }

    //update cluster parameters
    for(int k=0;k<K;k++)
      clusters.clear();
    for(int n=0;n<N;n++)
      clusters[assignments[n]].add(data[n][:]);
    if(debug)
      save(clusters,assignments)
  }

}
