#include <iostream>
#include <fstream>
using namespace std;

#define N 10000
#define D 3072
#define K 10

int main() {
    int label;
    char label_buffer[1];
    char pixel_buffer[D];

    ifstream binary;
    binary.open("data_batch_1.bin", ios::in | ios::binary);

    ofstream data;
    data.open("cifar.train");

    data << N << " " << D << " " << K << endl;

    for (int i = 0; i < N; i++) {
        binary.read(label_buffer, 1);
        label = (int)label_buffer[0];
        if (label < 0 || label >= K) {
            cout << "Warning: image " << i << " has label " << label << endl;
            return 1;
        }
        data << label;
        binary.read(pixel_buffer, D);
        for (int j = 0; j < D; j++) {
            data << " " << (int)(unsigned char)pixel_buffer[j];
        }
        data << endl;
    }

    data.close();
    binary.close();
    return 0;
}
