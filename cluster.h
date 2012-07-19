#ifndef M_CLUSTER
#define M_CLUSTER

#include <complex>
#include <iostream>

#include <Dense>
#include <Householder>
#include <Cholesky>

#include "distributions.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::HouseholderQR;

const bool USE_CACHING = true;
const double THETA = 2.0;

// consider changing to static member of Cluster?
int num_clusters = 0;

class Cluster {
  protected:
    int d, n;
    VectorXd sum;
    MatrixXd sum_squared;

    // Cached decompositions of the covariance matrix
    bool mu_is_cached;
    bool sigma_is_cached;
    bool inverse_is_cached;
    bool cholesky_is_cached;
    bool logdet_is_cached;
    VectorXd cached_mu;
    MatrixXd cached_sigma;
    MatrixXd cached_inverse;
    MatrixXd cached_cholesky;
    double cached_logdet;

  public:
    Cluster(int d) : d(d) {
        init();
    }

    void init() {
        clear_cache();
        n = 0;
        sum = VectorXd::Zero(d);
        sum_squared = 1e-6*MatrixXd::Identity(d,d);
    }

    void clear() {
        if (n > 0) {
            num_clusters--;
        }
        init();
    }

    void clear_cache() {
        mu_is_cached = false;
        sigma_is_cached = false;
        inverse_is_cached = false;
        cholesky_is_cached = false;
        logdet_is_cached = false;
    }

    int get_d() {
        return d;
    }

    int get_n() {
        return n;
    }

    void add(const VectorXd& x) {
        clear_cache();
        if (n == 0) {
            num_clusters++;
        }

        n++;
        sum += x;
        sum_squared += x*x.transpose();
    }

    void remove(const VectorXd& x) {
        clear_cache();
        if (n == 1) {
            num_clusters--;
        }

        n--;
        sum -= x;
        sum_squared -= x*x.transpose();
    }

    // Different distributions have different 'effective mu and sigma',
    // so this method does NOT necessarily return the mean and
    // covariance of points in the cluster.
    virtual VectorXd recompute_mu() = 0;
    virtual MatrixXd recompute_sigma() = 0;

    // Cached in cached_mu.
    const VectorXd& mu() {
        if (!mu_is_cached) {
            cached_mu = recompute_mu();
            mu_is_cached = USE_CACHING;
        }
        return cached_mu;
    }

    // Cached in cached_sigma.
    const MatrixXd& sigma() {
        if (!sigma_is_cached) {
            cached_sigma = recompute_sigma();
            sigma_is_cached = USE_CACHING;
        }
        return cached_sigma;
    }

    // Cached in cached_inverse.
    const MatrixXd& sigma_inverse() {
        if (!inverse_is_cached) {
            cached_inverse = sigma().inverse();
            inverse_is_cached = USE_CACHING;
        }
        return cached_inverse;
    }

    // Returns upper triangular U such that U^TU = sigma^-1.
    // Cached in cached_cholesky.
    const MatrixXd& cholesky() {
        if (!cholesky_is_cached) {
            cached_cholesky = sigma_inverse().llt().matrixU();
            cholesky_is_cached = USE_CACHING;
        }
        return cached_cholesky;
    }

    // Returns the log of the absolute value of the determinant of sigma().
    double logdet() {
        if (!logdet_is_cached) {
            HouseholderQR<MatrixXd> qr(sigma());
            cached_logdet = qr.logAbsDeterminant();
            logdet_is_cached = USE_CACHING;
        }
        return cached_logdet;
    }

    // O(d^2) to compute the logpdf
    virtual double log_pdf_norm(double norm_sq) = 0;
    virtual double log_pdf(const VectorXd & x) {
        VectorXd y = x - mu();
        return log_pdf_norm(y.dot(sigma_inverse() * y));
    }

    virtual double log_posterior_norm(double norm_sq) = 0;
    virtual double log_posterior(const VectorXd& x) {
        VectorXd y = x - mu();
        return log_posterior_norm(y.dot(sigma_inverse() * y));
    }
};

#endif
