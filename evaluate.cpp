#include <stdio.h>
#include <vector>
#include <assert.h>
#include <ctime>

#include "em.h"

#include <Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

int main(int argc, char **argv) {
    char filename[1000];
    assert(argc >= 2);
    sprintf(filename,"%s.train",argv[1]);
    printf("Reading training data...\n");
    FILE *ftrain = fopen(filename,"r");
    int N, D, K;
    fscanf(ftrain,"%d%d%d",&N,&D,&K);
    MatrixXd data_train = MatrixXd::Zero(N, D);
    vector<int> labels_train(N, -1);
    double temp_f; int temp_i;
    for(int n=0;n<N;n++){
        for(int d=0;d<D;d++){
            fscanf(ftrain,"%f",&temp_f);
            data_train(n,d) = temp_f;
        }
        fscanf(ftrain,"%d",&temp_i);
        labels_train[n] = temp_i;
    }
    printf("Done reading training data, starting training...\n");
    long long int time1 = clock();
    pair<vector<int>, vector<Cluster*> > em_output = em(data_train, K, &labels_train, -1, false, 1);
    long long int time2 = clock();
    printf("Done training: took %.4f seconds.\n", (time2-time1)/(float)CLOCKS_PER_SEC);
    fclose(ftrain);

    printf("Reading holdout data...\n");
    sprintf(filename,"%s.holdout",argv[1]);
    FILE *fholdout = fopen(filename,"r");
    int N2, D2, K2;
    fscanf(fholdout,"%d%d%d",&N2,&D2,&K2);
    assert(D2==D);
    assert(K2==K);
    MatrixXd data_holdout = MatrixXd::Zero(N2,D2);
    vector<int> labels_holdout = vector<int>(N2,-1);
    for(int n=0;n<N2;n++){
        for(int d=0;d<D2;d++){
            fscanf(fin,"%f",&temp_f);
            data_holdout(n,d)=temp_f;
        }
        fscanf(fin,"%d",&temp_i);
        labels_holdout[n]=temp_i;
    }
    printf("Done reading holdout data, starting predictions...\n");
    long long int time3 = clock();
    vector<int> labels_holdout_predicted = reassign_jl(em_output.second, data_holdout, vector<int>(N2,-1), 1);
    long long int time4 = clock();
    printf("Done assigning predictions: took %.4f seconds.\n", (time4-time3)/(float)CLOCKS_PER_SEC);

    int num_correct = 0;
    for(int n=0;n<N2;n++){
        if(labels_holdout_predicted[n] == labels_holdout[n]){
            num_correct++;
        }
    }
    printf("Got %d/%d correct.\n",num_correct,N2);

    return 0;
}
