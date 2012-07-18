#ifndef M_JLPROJECTION
#define M_JLPROJECTION
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

#include <Dense>

#include "calculations.h"
#include "distributions.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

const double EPS = 1e-9;
const double ACCEPT_MULTIPLIER = 0.98;

int get_proj_dim(int n, int d, int k) {
    return 10;
}

struct JLProjection {
    int m, d;

    // if x is d-dimensional, then |vecs_md * x| should be a good
    // estimate of |x|
    MatrixXd vecs_md;

    vector<ClusterStats *> clusters;
    vector<MatrixXd> projections_md;

    JLProjection(int _m, int _d) :
        m(_m), d(_d) {
        vecs_md = MatrixXd(m, d);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < d; j++)
                vecs_md(i, j) = gaussian() / sqrt(m);
        }
    }

    void add_cluster(ClusterStats * cluster) {
        clusters.push_back(cluster);
        // if U = cluster->cholesky(), then estimating x^t sigma^-1 x
        // is the same as estimating |Ux|^2
        MatrixXd proj = vecs_md * cluster->cholesky();
        //proj *= sqrt(d) / sqrt(m);
        projections_md.push_back(vecs_md * cluster->cholesky());
    }

    int assign_cluster(VectorXd x_d) {
        vector<double> est_loglikelies;
        vector<double> est_probs;

        for (int i = 0; i < (int)clusters.size(); i++) {
            VectorXd x_m = projections_md[i] * (x_d - clusters[i]->mu());
            VectorXd y_d = x_d - clusters[i]->mu();
            //cout << "y_d: " << y_d.dot(clusters[i]->sigma().inverse()*y_d) << ", x_m: " << x_m.dot(x_m) << endl;
            //getchar();
            double p = gaussian_logpdf(x_m.dot(x_m), d);
            p -= 0.5 * clusters[i]->log_abs_det();
            est_loglikelies.push_back(p);
        }

        // renormalize
        double max_log = est_loglikelies[0];
        for (int i = 0; i < (int)est_loglikelies.size(); i++)
            max_log = max(est_loglikelies[i], max_log);

        double sum_prob = 0.0;
        for (int i = 0; i < (int)est_loglikelies.size(); i++) {
            double prob = exp(est_loglikelies[i] - max_log);
            prob *= clusters[i]->n + THETA;
            est_probs.push_back(prob);
            sum_prob += prob;
        }

        while (true) {
            double t = random_double(sum_prob);
            int cur = 0;

            while (t > est_probs[cur]) {
                t -= est_probs[cur];
                cur++;
            }

            assert(cur < (int)clusters.size());

            double log_discrep = clusters[cur]->logpdf(x_d)
                - est_loglikelies[cur] + max_log;
            // TODO: eventually check that the discrepancy is not large
            // assert(abs(log_discrep) < SOMETHING);

            double accept = ACCEPT_MULTIPLIER * exp(log_discrep);
            /*cout << "est_loglikelies:" << endl << "  ";
            for (int i = 0; i < (int)est_loglikelies.size(); i++){
                cout << est_loglikelies[i] - max_log << " ";
            }
            cout << endl;*/
            //cout << "accept probability " << accept << endl;
            if (random_double() < accept)
                return cur;
        }

        assert(false);
    }
};

#endif
