#ifndef M_DISTRIBUTIONS
#define M_DISTRIBUTIONS
#include <ctime>
#include <cstdlib>
#include <cmath>

bool RAND_INITIALIZED = false; // maybe we shouldn't have this global...

void init_rand() {
  if (!RAND_INITIALIZED) {
    RAND_INITIALIZED = true;
    srand(time(NULL));
  }
}

// returns uniformly random real in [0, max]
double rand(double max_val) {
  init_rand();
  return rand() / (double) RAND_MAX;
}

// returns uniformly random integer in the interval [0, x)
int randint(int x) {
  init_rand();
  return rand() % x;
}

double gaussian() {
  double u1 = 1 - rand(1.0), u2 = 1 - rand(1.0);
  return sqrt(-2 * log(u1)) * cos(2 * M_PI * u2);
}

double gaussian(double mean, double stddev) {
  double g = gaussian();
  g *= stddev;
  g += mean;
  return g;
}

#endif
