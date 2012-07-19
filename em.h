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

void save(vector<Cluster*>& clusters, vector<int>* assignments=NULL) {
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

int reassign_naive(const MatrixXd& data, const vector<Cluster*>& clusters,
                   const vector<int>& pre_assignments, int S, vector<int>* assignments) {
    if (assignments == NULL) {
        return 1;
    }
    assignments->clear();

    int N = data.rows();
    int K = clusters.size();

    for (int n = 0; n < N; n++) {
        if(pre_assignments[n] != -1){
            for(int s = 0; s < S; s++){
                assignments->push_back(pre_assignments[n]);
            }
            continue;
        }
        vector<double> logprobs;
        for (int k = 0; k < K; k++) {
            logprobs.push_back(clusters[k]->log_posterior(data.row(n)));
        }
        for(int s = 0; s < S; s++){
            assignments->push_back(sample(logprobs));
        }
    }

    return 0;
}


int reassign_jl(const MatrixXd& data, const vector<Cluster*>& clusters,
                const vector<int>& pre_assignments, int S, vector<int>* assignments) {
    if (assignments == NULL) {
        return 1;
    }
    assignments->clear();

    int N = data.rows();
    int K = clusters.size();
    int D = data.cols();

    JLProjection jlp(get_proj_dim(N, D, K), D);

    for (int k = 0; k < K; k++)
        jlp.add_cluster(clusters[k]);

    for (int n = 0; n < N; n++){
        if(pre_assignments[n] != -1){
            for(int s = 0; s < S; s++){
                assignments->push_back(pre_assignments[n]);
            }
            continue;
        }
        for(int s = 0; s < S; s++){
            assignments->push_back(jlp.assign_cluster(data.row(n)));
        }
    }

    return 0;
}


int em(const MatrixXd& data, int K, const vector<int>& pre_assignments,
       vector<int>* assignments_ptr, vector<Cluster*>* clusters_ptr, \
       int T=-1, bool debug=false, int S=1) {
    if (assignments_ptr == NULL || clusters_ptr == NULL) {
        return 1;
    }
    vector<int>& assignments = *assignments_ptr;
    vector<Cluster*>& clusters = *clusters_ptr;
    assignments.clear();
    clusters.clear();

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

    for (int k = 0; k < K; k++){
        // TODO: maybe add a small multiple of identity to total_covar
        Cluster* new_cluster = new TCluster(
            D, D + 2.0, 0.1, total_mean, total_covar);
        clusters.push_back(new_cluster);
    }
    for(int n = 0; n < N; n++){
        if(pre_assignments[n] != -1){
            clusters[pre_assignments[n]]->add(data.row(n));
        }
    }
    for(int k = 0; k < K; k++){
        if(clusters[k]->get_n() == 0){
            clusters[k]->add(data.row(random_int(N)));
        }
    }
    if (debug) {
        save(clusters);
    }

    double loglikelihood_old;
    for(int t = 0; t != T; t++) {
        if (reassign_jl(data, clusters, pre_assignments, S, &assignments) != 0) {
            return 1;
        }

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
            double log_posteriors[S];
            for (int s = 0; s < S; s++)
                log_posteriors[s] = clusters[assignments[S*n + s]]->log_posterior(data.row(n));

            double max_log = log_posteriors[0];
            for (int s = 0; s < S; s++)
                max_log = max(max_log, log_posteriors[s]);
            loglikelihood += max_log;

            double prob = 0.0;
            for (int s = 0; s < S; s++)
                prob += exp(log_posteriors[s] - max_log);
            prob /= S;

            loglikelihood += log(prob);
        }
        //cout << loglikelihood << endl;

        if (t == T - 1 || (T == -1 && t>0 && loglikelihood < loglikelihood_old + 1e-4)) {
            cout << "[";
            for (int n = 0; n < N; n++) {
                // TODO: handle soft assignments?
                cout << assignments[S*n] << (n == N - 1 ? "" : ",");
            }
            cout << "]" << endl;
            break;
        }

        loglikelihood_old = loglikelihood;


        if (debug) {
            save(clusters, &assignments);
        }

    }
    return 0;
}
