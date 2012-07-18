#ifndef M_GAUSSIAN_CLUSTER
#define M_GAUSSIAN_CLUSTER

#include "cluster.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::HouseholderQR;

struct GaussianCluster : public Cluster {
  public:
    GaussianCluster(int d) : Cluster(d) {
        clear();
    }

    VectorXd recompute_mu() {
        return sum / n;
    }

    MatrixXd recompute_sigma() {
        return sum_squared / n - sum * sum.adjoint() / n / n;
    }

    // O(d^2) to compute the logpdf, except for computing the
    // determinant and inverse. These are O(d^3), but cached.
    double log_pdf_norm(double norm_sq) {
        return -0.5*d*log(2*M_PI) - 0.5* \
               - 0.5*norm_sq;
    }

    double log_posterior_norm(double norm_sq) {
        return log_pdf_norm(norm_sq) + log(n + THETA);
    }
};

#endif
