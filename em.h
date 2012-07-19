#include <fstream>
#include <vector>
#include <cmath>
#include <assert.h>

#include <Dense>

#include "t_cluster.h"
#include "gaussian_cluster.h"
#include "distributions.h"
#include "jlprojection.h"

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

void save(vector<Cluster*> & clusters, vector<int>* assignments=NULL) {
    int K = clusters.size();
    int N;
    if(assignments == NULL){
        N = -1;
    }
    else {
        N = assignments->size();
    }
    int D = clusters[0]->get_d();
//    fout << (assignments == NULL ? 0 : 1) << endl;
//    fout << N << " " << D << " " << K << endl;
    fout << (assignments == NULL ? "Log type 0: initial_clusters" :
            "Log type 1: current_iteration") << endl;
    fout << "N: " << N << ", D: " << D << ", K: " << K << endl;
    for (int k = 0; k < K; k++) {
        fout << "Cluster mu and sigma:" << endl;
        fout << clusters[k]->mu() << endl;
        fout << clusters[k]->sigma() << endl;
    }
    if (assignments != NULL) {
        fout << "Assignments:" << endl;
        for(int n = 0; n < N; n++) {
            fout << (*assignments)[n] << endl;
        }
    }
}

vector<int> reassign_naive(vector<Cluster*> & clusters, const MatrixXd & data, int S) {
    int N = data.rows();
    int K = clusters.size();

    vector<int> assignments;
    for (int n = 0; n < N; n++) {
        vector<double> logprobs;
        for (int k = 0; k < K; k++) {
            logprobs.push_back(clusters[k]->log_posterior(data.row(n)));
        }
        for(int s = 0; s < S; s++){
            assignments.push_back(sample(logprobs));
        }
    }

    return assignments;
}


vector<int> reassign_jl(vector<Cluster*> & clusters, const MatrixXd & data, int S) {
    int N = data.rows();
    int K = clusters.size();
    int D = data.cols();

    vector<int> assignments;
    JLProjection jlp(get_proj_dim(N, D, K), D);

    for (int i = 0; i < (int)clusters.size(); i++)
        jlp.add_cluster(clusters[i]);

    for (int i = 0; i < N; i++){
        for(int s = 0; s < S; s++){
            assignments.push_back(jlp.assign_cluster(data.row(i)));
        }
    }

    return assignments;
}


void em(MatrixXd data, int K, int T=-1, bool debug=false, int S=1) {
    int N = data.rows();
    int D = data.cols();
    if (debug) {
        save_init(data);
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

    vector<Cluster*> clusters;
    for (int k = 0; k < K; k++){
        // TODO: maybe add a small multiple of identity to total_covar
        Cluster* new_cluster = new TCluster(
            D, D + 2.0, 0.1, total_mean, total_covar);
        new_cluster->add(data.row(random_int(N)));
        clusters.push_back(new_cluster);
    }
    if (debug) {
        save(clusters);
    }

    double loglikelihood_old;
    for(int t = 0; t != T; t++) {
        vector<int> assignments = reassign_naive(clusters, data, S);

        //update cluster parameters
        for (int k = 0; k < K; k++) {
            clusters[k]->clear();
        }
        for (int n = 0; n < N; n++) {
            for(int s = 0; s < S; s++){
                clusters[assignments[S*n+s]]->add(data.row(n));
            }
        }

        //compute loglikelihood to test convergence
        double loglikelihood = 0.0;
        for(int n = 0; n < N; n++){
            for(int s = 0; s < S; s++){
                loglikelihood += clusters[assignments[S*n+s]]->log_posterior(data.row(n)) / S;
            }
        }
        cout << loglikelihood << endl;
        if(T == -1 && t>0 && loglikelihood < loglikelihood_old + 1e-4)
            break;
        loglikelihood_old = loglikelihood;


        if (debug) {
            save(clusters, &assignments);
        }
    }
}
