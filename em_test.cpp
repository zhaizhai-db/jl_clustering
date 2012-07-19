#include <iostream>
#include <fstream>
#include <vector>

#include "em.h"
#include "t_cluster.h"

#include <Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

void test1(){
  MatrixXd data(5,2);
  data << 1.0,0.0,
          1.0,0.1,
          1.0,0.2,
          5.0,2.0,
          5.0,2.1;
  em(data,2,5,true);
}

void test2(){
    ifstream fin("data.in");
    int N,D,K;
    fin >> N >> D >> K;
    MatrixXd data(N,D);
    for(int n=0;n<N;n++){
        for(int d=0;d<D;d++){
            double x;
            fin >> x;
            cout << "x=" << x << endl;
            data(n,d)=x;
        }
    }
    em(data,K,20,true,1);
}

void libras_test(){
    ifstream fin("modified_libras.data");
    int N,D,K;
    fin >> N >> D >> K;
    MatrixXd data(N,D);
    vector<int> labels;
    for(int n=0;n<N;n++){
        for(int d=0;d<D;d++){
            double x;
            fin >> x;
            data(n,d)=x;
        }
        int l;
        fin >> l;
        labels.push_back(l);
        //cout << "label=" << l << endl;
    }

    VectorXd total_mean = VectorXd::Zero(D);
    MatrixXd total_covar = MatrixXd::Zero(D, D);
    for (int i = 0; i < N; i++) {
        VectorXd x = data.row(i);
        total_mean += x;
        total_covar += x * x.transpose();
    }
    total_mean /= N;
    total_covar /= N;

    vector<TCluster> clusters;
    for (int k = 0; k < K; k++)
        clusters.push_back(TCluster(D, D + 2.0, 0.1, total_mean, total_covar));

    for (int i = 0; i < N; i++)
        clusters[labels[i] - 1].add(data.row(i));

    double log_posterior = 0.0;
    for (int i = 0; i < N; i++)
        log_posterior += clusters[labels[i] - 1].log_posterior(data.row(i));

    //cout << "log posterior from true labels: " << log_posterior << endl;

    em(data,K,-1,true);
}

int main() {
    //test1();
    //test2();
    libras_test();
}
