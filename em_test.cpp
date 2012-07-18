#include <iostream>
#include <fstream>
#include <vector>

#include "em.h"

#include <Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

void test1(){
  MatrixXd data(5,2);
  data << 1.0,0.0,
          1.0,0.1,
          1.0,0.2,
          5.0,2.0,
          5.0,2.1;
  em(data,2,5,true);
}

void test2(){
    ifstream fin("data.in");
    int N,D,K;
    fin >> N >> D >> K;
    MatrixXd data(N,D);
    for(int n=0;n<N;n++){
        for(int d=0;d<D;d++){
            double x;
            fin >> x;
            cout << "x=" << x << endl;
            data(n,d)=x;
        }
    }
    em(data,K,-1,true,1);
}

void libras_test(){
    ifstream fin("modified_libras.data");
    int N,D,K;
    fin >> N >> D >> K;
    MatrixXd data(N,D);
    vector<int> labels;
    for(int n=0;n<N;n++){
        for(int d=0;d<D;d++){
            double x;
            fin >> x;
            data(n,d)=x;
        }
        int l;
        fin >> l;
        labels.push_back(l);
        cout << "label=" << l << endl;
    }
    em(data,K,50,true);
}

int main() {
    //test1();
    test2();
    //libras_test();
}
