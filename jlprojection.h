#ifndef M_JLPROJECTION
#define M_JLPROJECTION

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

#include <Dense>

#include "cluster.h"
#include "distributions.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

int get_proj_dim(int n, int d, int k) {
    return 100;
}

double compute_min_bound(int m, double error_prob) {
    double x = 1.0 / (1.0 + sqrt(-2 * log(error_prob) / m));
    return x * x;
}

struct JLProjection {
    int m, d;
    int num_tries, num_calls;

    // if x is d-dimensional, then |vecs_md * x| should be a good
    // estimate of |x|
    MatrixXd vecs_md;

    // if y = vecs_md * x, then we want |y|^2 >= min_bound * |x|^2 w.h.p.
    double min_bound;

    vector<Cluster *> clusters;
    vector<MatrixXd> projections_md;

    JLProjection(int _m, int _d, double error_prob=1e-6) :
        m(_m), d(_d), num_tries(0), num_calls(0) {
        vecs_md = MatrixXd(m, d);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < d; j++){
                vecs_md(i, j) = (2.0*random_int(2)-1) / sqrt(m);
            }
        }
        min_bound = compute_min_bound(m, error_prob);
    }

    void add_cluster(Cluster * cluster) {
        clusters.push_back(cluster);
        // TODO fix this documentation
        // if U = cluster->cholesky(), then estimating x^t sigma^-1 x
        // is the same as estimating |Ux|^2
        MatrixXd proj = cluster->cholesky2().triangularView<Lower>().solve<OnTheRight>(vecs_md);
        projections_md.push_back(proj);
    }

    void reset_num_tries(){
        num_tries = 0;
        num_calls = 0;
    }
    double avg_calls(){
        //assert(num_calls != 0);
        if(num_calls == 0) return -1.0;
        return num_tries/(float)num_calls;
    }

    int assign_cluster(VectorXd x_d) {
        vector<double> est_loglikelies;

        for (int i = 0; i < (int)clusters.size(); i++) {
            VectorXd x_m = projections_md[i] * (x_d - clusters[i]->mu());
            est_loglikelies.push_back(
                clusters[i]->log_posterior_norm(min_bound * x_m.dot(x_m)));
        }

        num_calls++;
        while (true) {
            int prop = sample_log(est_loglikelies);
            double true_logprob = clusters[prop]->log_posterior(x_d);
            num_tries++;

            double accept = exp((true_logprob - est_loglikelies[prop])/ANNEAL_TEMPERATURE);
            if(accept > 1.0) printf("accept: %f\n", accept);
            assert(accept <= 1.0);
            if (random_double() < accept){
                //printf("Evaluated log_posterior %d times.\n",num_tries);
                return prop;
            }

            est_loglikelies[prop] = true_logprob;
        }

        assert(false);
    }
};

#endif
