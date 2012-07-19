#ifndef M_T_CLUSTER
#define M_T_CLUSTER

#include <Eigenvalues>

#include "cluster.h"

using Eigen::EigenSolver;

const double ALPHA = 0.5;

struct TCluster : public Cluster {
  private:
    double v0, k0;
    VectorXd u0;
    MatrixXd T0;

  public:
    TCluster(int d, double v0, double k0, const VectorXd& u0, const MatrixXd& T0) :
            Cluster(d), v0(v0), k0(k0), u0(u0), T0(T0) {
        clear();
    }

    // O(d^2) to compute all the parameters.
    // O(d^3) to compute inverse and determinant, but these are cached.
    //
    // Equations for parameters:
    // x ~ t_{v_n-d+1}(u_n, T_n(k_n+1)/k_n(v_n-d+1))
    // v_n = v0 + n
    // k_n = k0 + n
    // u_n = (k*u_0 + n*xbar)/(k+n)
    // T_n = T0 + S_n + k*n/(k+n)*(u0-xbar)*(u0-xbar)^T
    // S_n = sum (x_i-xbar)*(x_i-xbar)^T
    //
    // Equations for the actual computation:
    // t(x) = Gamma(v/2+d/2)/Gamma(v/2) * det(Sigma)^-0.5
    // / [(pi*v)^(d/2)] * [1+(1/v)*(x-u)^T*Sigma^-1*(x-u)]^-((v+d)/2)
    //
    // To complete all of these it is sufficient to store:
    // n
    // sum x_i
    // sum x_i*x_i^T
    VectorXd recompute_mu() {
        if (n > 0) {
            return (k0*u0 + sum)/(k0 + n);
        }
        return u0;
    }

    MatrixXd recompute_sigma() {
        if (n > 0) {
            VectorXd xbar = sum / n;
            MatrixXd covar = sum_squared / n - xbar * xbar.transpose();
            MatrixXd Tn = T0 + n*covar + k0*n/(k0 + n)*(u0 - xbar)*(u0 - xbar).transpose();
            return Tn*(k0 + n + 1)/((k0 + n)*(v0 + n - d + 1));
        }
        return T0*(k0 + 1)/(k0*(v0 - d + 1));
    }

    double log_pdf_norm(double norm_sq) {
        double v = v0 + n - d + 1;
        return gammaln((v + d)/2.0) - gammaln(v/2.0) \
               - 0.5*logdet() - 0.5*d*log(M_PI*v) \
               - 0.5*(v + d)*log(1 + (1.0/v)*norm_sq);
    }

    double log_posterior_norm(double norm_sq) {
        return log_pdf_norm(norm_sq) + (n == 0 ? log(THETA + ALPHA*num_clusters) : log(n - ALPHA));
    }
};

#endif
