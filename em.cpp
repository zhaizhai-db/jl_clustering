#include <fstream>
#include <vector>
#include "calculations.h"

#include <Dense>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

ofstream fout("log.txt");

void save_init(MatrixXd data) {
    fout << 2 << endl;
    fout << data.rows() << data.cols() << endl;
    fout << data << endl;
}

void save(vector<ClusterStats> clusters, vector<int>* assignments=NULL) {
    int K = clusters.size();
    int N = clusters[0].mu().size();
    int D = clusters[0].d;
    fout << (assignments == NULL ? 0 : 1) << endl;
    fout << N << D << K << endl;
    for (int k = 0; k < K; k++) {
        fout << clusters[k].mu() << endl;
        fout << clusters[k].sigma() << endl;
    }
    if (assignments != NULL) {
        for(int n = 0; n < N; n++) {
            fout << (*assignments)[n] << endl;
        }
    }
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
        new_cluster.add(data.row(rand()));
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
            clusters.clear();
        }
        for (int n = 0; n < N; n++) {
            clusters[assignments[n]].add(data.row(n));
        }
        if (debug) {
            save(clusters, &assignments);
        }
    }
}
