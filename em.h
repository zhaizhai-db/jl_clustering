#include <fstream>
#include <vector>
#include <cmath>
#include <ctime>
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

int ALGORITHM;
double K0, V0, LAMBDA;
void initialize_em(int _ALGORITHM, double _K0, double _V0, double _LAMBDA){
    ALGORITHM=_ALGORITHM;
    assert(ALGORITHM == 0 || ALGORITHM == 1); // 0 = naive, 1 = jl
    K0=_K0;
    assert(K0 > 0);
    V0=_V0;
    assert(V0 > 0); // TODO check this is sufficient
    LAMBDA=_LAMBDA;
    assert(0.0 <= LAMBDA && LAMBDA <= 1.0);
}

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

double magnitude(const VectorXd& x) {
    return x.transpose()*x;
}

int kmeans(const MatrixXd& data, int K, const vector<int>& pre_assignments,
           int S, vector<int>* assignments) {
    if (assignments == NULL) {
        return 1;
    }
    assignments->clear();

    int N = data.rows();
    int D = data.cols();

    // Compute the means of the pre-assigned clusters
    VectorXd* means = new VectorXd[K];
    int* sizes = new int[K];
    for (int k = 0; k < K; k++) {
        means[K] = VectorXd::Zero(D);
    }
    for (int n = 0; n < N; n++) {
        if (pre_assignments[n] != -1) {
            int k = pre_assignments[n];
            means[k] += data.row(n);
            sizes[k] += 1;
        }
    }
    for (int k = 0; k < K; k++) {
        means[k] /= sizes[k];
    }

    for (int n = 0; n < N; n++) {
        if(pre_assignments[n] != -1){
            for(int s = 0; s < S; s++){
                assignments->push_back(pre_assignments[n]);
            }
            continue;
        }
        float best_dist = magnitude(data.row(n) - means[0]);
        int best_index = 0;
        for (int k = 1; k < K; k++) {
            float dist = magnitude(data.row(n) - means[k]);
            if (dist < best_dist) {
                dist = best_dist;
                best_index = k;
            }
        }
        for(int s = 0; s < S; s++){
            assignments->push_back(best_index);
        }
    }

    delete[] means;
    delete[] sizes;
    return 0;
}

int reassign_naive(const MatrixXd& data, const vector<Cluster*>& clusters,
                   const vector<int>& pre_assignments, int S, vector<int>* assignments) {
    if (assignments == NULL) {
        return 1;
    }
    assignments->clear();

    int N = data.rows();
    int K = clusters.size();

    cout << "Computing new assignments..." << flush;
    long long int time1 = clock();
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
    long long int time2 = clock();
    printf(" Done. Time elapsed: %.4f seconds.\n", (time2-time1)/(float)CLOCKS_PER_SEC);

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
    long long int time1, time2;

    cout << "forming JL projectors..." << flush;
    time1 = clock();
    JLProjection jlp(get_proj_dim(N, D, K), D);
    //jlp.reset_num_tries();
    time2 = clock();
    printf("Done. Time elapsed: %.4f seconds.\n", (time2-time1)/(float)CLOCKS_PER_SEC);

    cout << "initializing JL clusters (requires cholesky)..." << flush;
    time1 = clock();
    for (int k = 0; k < K; k++)
        jlp.add_cluster(clusters[k]);
    time2 = clock();
    printf("Done. Time elapsed: %.4f seconds.\n", (time2-time1)/(float)CLOCKS_PER_SEC);

    cout << "computing new assignments..." << flush;
    time1 = clock();
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
    time2 = clock();
    printf("Done. Time elapsed: %.4f seconds.\n", (time2-time1)/(float)CLOCKS_PER_SEC);
    printf("Average number of calls per point: %.4f\n", jlp.avg_calls());

    return 0;
}


int em(const MatrixXd& data, int K, const vector<int>& pre_assignments,
       vector<int>* assignments_ptr, vector<Cluster*>* clusters_ptr, \
       int T=-1, bool debug=false, int S=1, bool use_kmeans=false) {
    if (assignments_ptr == NULL || clusters_ptr == NULL) {
        return 1;
    }
    vector<int>& assignments = *assignments_ptr;
    vector<Cluster*>& clusters = *clusters_ptr;
    assignments.clear();
    clusters.clear();

    int N = data.rows();
    int D = data.cols();
    long long int time1, time2;
    cout << "N: " << N << ", D: " << D << endl;

    if (debug) {
        cout << "Saving initial data for debugging..." << flush;
        save_init(data);
        cout << " Done." << endl;
    }

    cout << "Computing mean and covariance of data set..." << flush;
    time1 = clock();
    VectorXd total_mean = VectorXd::Zero(D);
    MatrixXd total_covar = MatrixXd::Zero(D, D);
    for (int i = 0; i < N; i++) {
        VectorXd x = data.row(i);
        total_mean += x;
        total_covar += x * x.transpose();
    }
    total_mean /= N;
    total_covar /= N;
    total_covar -= total_mean * total_mean.transpose();
    total_covar = (1.0-LAMBDA) * total_covar + LAMBDA * total_covar.trace() * MatrixXd::Identity(D, D);
    time2 = clock();
    printf(" Done. Time elapsed: %.4f seconds.\n", (time2-time1)/(float)CLOCKS_PER_SEC);

    for (int k = 0; k < K; k++){
        // TODO: maybe add a small multiple of identity to total_covar
        Cluster* new_cluster = new TCluster(
            D, D + V0, K0, total_mean, total_covar);
        clusters.push_back(new_cluster);
    }
    for(int n = 0; n < N; n++){
        if(pre_assignments[n] != -1){
            clusters[pre_assignments[n]]->add(data.row(n));
        }
    }
    assignments = pre_assignments;
    for(int k = 0; k < K; k++){
        assert(clusters[k]->get_n() != 0);
        /*if(clusters[k]->get_n() == 0){
            clusters[k]->add(data.row(random_int(N)));
        }*/
    }
    if (debug) {
        save(clusters);
    }

    if (use_kmeans) {
        cout << "Running kmeans precomp." << endl;
        kmeans(data, K, pre_assignments, S, &assignments);
    }

    double loglikelihood_old = -1e9;
    for(int t = 0; t != T; t++) {
        cout << "Starting iteration " << t << "." << endl;
        vector<int> assignments_old = assignments;
        int status;
        if(ALGORITHM == 0){
            status = reassign_naive(data, clusters, pre_assignments, S, &assignments);
        }
        else if(ALGORITHM == 1) {
            status = reassign_jl(data, clusters, pre_assignments, S, &assignments);
        } else {
            status = 1;
        }
        if(status != 0) {
            return 1;
        }
        //cout << "Done computing assignments" << endl;
        //update cluster parameters
        /*for (int k = 0; k < K; k++) {
            clusters[k]->clear();
        }*/
        cout << "Updating clusters..." << flush;
        time1 = clock();
        for (int n = 0; n < N; n++) {
            for(int s = 0; s < S; s++){
                if(assignments_old[S*n+s] != assignments[S*n+s]){
                    if(assignments_old[S*n+s] != -1)
                        clusters[assignments[S*n+s]]->remove(data.row(n));
                    if(assignments[S*n+s] != -1)
                        clusters[assignments[S*n+s]]->add(data.row(n));
                }
            }
        }
        time2 = clock();
        printf(" Done updating clusters. Time elapsed: %.4f seconds.\n", (time2-time1)/(float)CLOCKS_PER_SEC);

        //compute loglikelihood to test convergence
        double loglikelihood = 0.0;
        cout << "Computing loglikelihood..." << flush;
        time1 = clock();
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
        time2 = clock();
        printf("Done. Time elapsed: %.4f seconds.\n", (time2-time1)/(float)CLOCKS_PER_SEC);
        cout << "Log likelihood: " << loglikelihood << endl;

        if (t == T - 1 || (T < 0 && t>0 && loglikelihood < loglikelihood_old + 1e-4)) {
            /*cout << "[";
            for (int n = 0; n < N; n++) {
                // TODO: handle soft assignments?
                cout << assignments[S*n] << (n == N - 1 ? "" : ",");
            }
            cout << "]" << endl;*/
            break;
        }

        loglikelihood_old = loglikelihood;


        if (debug) {
            save(clusters, &assignments);
        }

    }
    return 0;
}
