#include <iostream>
#include <fstream>
using namespace std;

#define N (10000)
#define D (3072)
#define K (10)
#define T (5)

int main() {
    int label;
    char label_buffer[1];
    char pixel_buffer[D];

    ofstream data;
    data.open("cifar.pretrain");

    data << ((T-1)*N) << " " << D << " " << K << endl;
    char filename[1000];

    for(int t = 1; t <= T+1; t++){
        if(t == T){
            data.close();
            data.open("cifar.holdout");
            data << N << " " << D << " " << K << endl;
        } else if(t == T+1){
            data.close();
            data.open("cifar.test");
            data << N << " " << D << " " << K << endl;
        }
        ifstream binary;
        if(t <= T){
            sprintf(filename,"data_batch_%d.bin",t);
        } else {
            sprintf(filename,"test_batch.bin");
        }
        binary.open(filename, ios::in | ios::binary);

        for (int i = 0; i < N; i++) {
            binary.read(label_buffer, 1);
            label = (int)label_buffer[0];
            if (label < 0 || label >= K) {
                cout << "Warning: image " << i << " has label " << label << endl;
                return 1;
            }
            binary.read(pixel_buffer, D);
            for (int j = 0; j < D; j++) {
                data << (int)(unsigned char)pixel_buffer[j] << " ";
            }
            data << label;
            data << endl;
        }

        binary.close();
    }    
    data.close();
    return 0;
}
