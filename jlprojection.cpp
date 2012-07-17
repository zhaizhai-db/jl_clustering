#include <iostream>
#include <vector>

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
    // TODO: make it random
  }

  void add_cluster(ClusterStats * cluster) {
    clusters.push_back(cluster);
    // TODO: check if cluster->cholesky() is right
    projections_md.push_back(vecs_md * cluster->cholesky());
  }

  ClusterStats * assign_cluster(VectorXd x_d) {
    double sum_prob = 0.0;
    vector<double> est_probs;

    for (int i = 0; i < clusters.size(); i++) {
      VectorXd v_m = projections_md[i] * x_d;
      double p = v_m.dot(v_m);
      if (p < EPS)
        continue;
      est_probs.push_back(p);
      sum_prob += p;
    }

    assert(est_probs.size() > 0);

    while (true) {
      double t = rand(sum_prob);
      int cur = 0;

      while (t > est_probs[cur]) {
        t -= est_probs[cur];
        cur++;
      }

      // TODO: exp
      double accept = ACCEPT_MULTIPLIER
        * (exp(clusters[cur]->logpdf_em(x)) / est_probs[cur]);
      if (rand(1.0) < accept)
        return clusters[cur];
    }

    return NULL;
  }
};
