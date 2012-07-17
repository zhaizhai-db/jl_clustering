#include <iostream>
#include <vector>
#include <cmath>
#include <Dense>
#include "calculations.h"
#include "distributions.h"

using namespace Eigen;
using namespace std;

const double EPS = 1e-9;
const double ACCEPT_MULTIPLIER = 0.98;

struct JLProjection {
    int m, d;
    MatrixXd vecs_md;
    vector<ClusterStats *> clusters;
    vector<MatrixXd> projections_md;

    JLProjection(int _m, int _d) :
        m(_m), d(_d) {
        vecs_md = MatrixXd(m, d);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < d; j++)
                vecs_md(i, j) = gaussian() / d;
        }
    }

    void add_cluster(ClusterStats * cluster) {
        clusters.push_back(cluster);
        // TODO: check if cluster->cholesky() is right
        projections_md.push_back(vecs_md * cluster->cholesky());
    }

    ClusterStats * assign_cluster(VectorXd x_d) {
        vector<double> est_logprobs;

        for (int i = 0; i < clusters.size(); i++) {
            VectorXd v_m = projections_md[i] * x_d;
            double p = gaussian_logpdf(v_m.dot(v_m)); // TODO: implement
            est_logprobs.push_back(p);
        }

        // renormalize
        double max_log = est_logprobs[0];
        for (int i = 0; i < est_logprobs.size(); i++)
            max_log = max(est_logprobs[i], max_log);
        for (int i = 0; i < est_logprobs.size(); i++)
            est_logprobs[i] -= max_log;

        double sum_prob = 0.0;
        for (int i = 0; i < est_logprobs.size(); i++)
            sum_prob += exp(est_logprobs[i]) * clusters[i]->n;

        while (true) {
            double t = rand(sum_prob);
            int cur = 0;

            while (t > est_logprobs[cur]) {
                t -= exp(est_logprobs[cur]) * clusters[i]->n;
                cur++;
            }

            double log_discrep = clusters[cur]->logpdf(x_d)
                - est_logprobs[cur] + max_log;
            // TODO: eventually check that the discrepancy is not large
            // assert(abs(log_discrep) < SOMETHING);

            double accept = ACCEPT_MULTIPLIER * exp(log_discrep);
            if (rand(1.0) < accept)
                return clusters[cur];
        }

        return NULL;
    }
};

int main() {
}
