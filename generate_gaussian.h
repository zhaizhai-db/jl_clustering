#include <iostream>
#include <fstream>
#include <vector>

#include "distributions.h"

#include <Dense>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

void generate_data(const vector<GaussianDistribution> & sources,
                   const vector<double> & weights, int N, const char * outfile) {
    int K = sources.size();
    assert(K == (int) weights.size());

    int D = sources[0].dim;
    for (int i = 0; i < K; i++)
        assert(D == sources[i].dim);

    ofstream fout(outfile);
    fout << N << " " << D << " " << K << '\n';

    for (int i = 0; i < N; i++) {
        // O(K + D^2) per iteration
        int ix = sample(weights);

        VectorXd x = sources[ix].draw();
        for (int i = 0; i < D; i++) {
            fout << x(i) << " ";
        }
        fout << ix << '\n';
    }
}
