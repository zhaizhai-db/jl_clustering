#include <fstream>
#include <vector>
#include <cmath>
#include <assert.h>
#include "calculations.h"
#include "distributions.h"

#include <Dense>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

ofstream fout("log.txt");

void save_init(MatrixXd data) {
//    fout << 2 << endl;
//    fout << data.rows() << " " << data.cols() << endl;
//    fout << data << endl;
    fout << "Log type 2: data_matrix" << endl;
    fout << "Width: " << data.rows() << ", height: " << data.cols() << endl;
    fout << "Data:" << endl << data << endl;
}

void save(vector<ClusterStats> clusters, vector<int>* assignments=NULL) {
    int K = clusters.size();
    int N;
    if(assignments == NULL){
        N = -1;
    }
    else {
        N = assignments->size();
    }
    int D = clusters[0].d;
//    fout << (assignments == NULL ? 0 : 1) << endl;
//    fout << N << " " << D << " " << K << endl;
    fout << (assignments == NULL ? "Log type 0: initial_clusters" :
            "Log type 1: current_iteration") << endl;
    fout << "N: " << N << ", D: " << D << ", K: " << K << endl;
    for (int k = 0; k < K; k++) {
        fout << "Cluster " << k << " mu and sigma:" << endl;
        fout << clusters[k].mu() << endl;
        fout << clusters[k].sigma() << endl;
    }
    if (assignments != NULL) {
        fout << "Assignments of " << N << " points:" << endl;
        for(int n = 0; n < N; n++) {
            fout << (*assignments)[n] << endl;
        }
    }
}

int sample(vector<double> logprobs){
    double largest = logprobs[0];
    for(int i=1;i<(int)logprobs.size();i++){
        largest = max(largest,logprobs[i]);
    }
    double sum = 0.0;
    for(int i=0;i<(int)logprobs.size();i++){
        //cout << "logprobs[" << i << "]=" << logprobs[i] << endl;
        logprobs[i] -= largest;
        sum += exp(logprobs[i]);
    }
    double u = random_double(sum), partial_sum = 0.0;
    //cout << "u=" << u << " sum=" << sum << endl;
    for(int i=0;i<(int)logprobs.size();i++){
        partial_sum += exp(logprobs[i]);
        //cout << "partial sum=" << partial_sum << endl;
        if(u < partial_sum){
            return i;
        }
    }
    assert(false);
}

void em(MatrixXd data, int K, int T=-1, bool debug=false) {
    int N = data.rows();
    int D = data.cols();
    if (debug) {
        save_init(data);
    }

    vector<ClusterStats> clusters;
    for (int k = 0; k < K; k++){
        ClusterStats new_cluster(D, 0.0, 0.0, VectorXd(D), MatrixXd(D, D));
        new_cluster.add(data.row(random_int(N)));
        new_cluster.sum_squared += MatrixXd::Identity(D,D);
        clusters.push_back(new_cluster);
    }
    if (debug) {
        save(clusters);
    }

    for(int t = 0; t != T; t++) { //TODO deal with case when T == -1
        //update assignments
        vector<int> assignments;
        for (int n = 0; n < N; n++) {
            vector<double> logprobs;
            for (int k = 0; k < K; k++) {
                logprobs.push_back(clusters[k].logpdf_em(data.row(n)));
            }
            assignments.push_back(sample(logprobs));
        }

        //update cluster parameters
        for (int k = 0; k < K; k++) {
            clusters[k].clear();
        }
        for (int n = 0; n < N; n++) {
            clusters[assignments[n]].add(data.row(n));
        }
        if (debug) {
            save(clusters, &assignments);
        }
    }
}
