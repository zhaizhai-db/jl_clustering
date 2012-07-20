#include <cassert>
#include <iostream>
#include <fstream>
using namespace std;

// This dataset includes 24-bit RGB data for many 32x32 pixel
// images, so each image comprises 3072 bytes.
#define NUM_PIXELS (1024)

// Scaling downsamples D by a factor of SCALE^2, so SCALE
// should be a divisor of 32.
// In addition, turning off USE_COLOR downsamples D by 3x.
#define SCALE (1)
#define USE_COLOR (false)

// N can be any positive integer up to (and including) 10000.
// The training set will have (T-1)*N points, while the holdout
// and test data will have N points each.
#define N (10000)
#define D ((USE_COLOR ? 3 : 1)*NUM_PIXELS/(SCALE*SCALE))
#define K (10)
#define T (5)

int extract_pixel(char* pixels, int i, int scale=1, bool use_color=true) {
    if (!use_color) {
        return (extract_pixel(pixels, 3*i, scale) + extract_pixel(pixels, 3*i + 1, scale) + 
                + extract_pixel(pixels, 3*i + 2, scale))/3;
    } else if (scale == 1) {
        assert(0 <= i && i < 3*NUM_PIXELS);            
        return (int)(unsigned char)pixels[i];
    } else {
        assert(scale != 0 && 32 % scale == 0);
        int result = 0;
        int width = 32 / scale;
        int offset = (i % 3) + 3*((i/3) % width) + 3*32*((i / (3*width)));
        for (int j = 0; j < scale; j++) {
            for (int k = 0; k < scale; k++) {
                result += extract_pixel(pixels, 3*(32*j + k) + offset, 1);
            }
        }
        return result / scale / scale;
    }
}

int main() {
    int label;
    char label_buffer[1];
    char pixel_buffer[3*NUM_PIXELS];

    ofstream data;
    data.open("../cifar.pretrain");

    data << ((T-1)*N) << " " << D << " " << K << endl;
    char filename[1000];

    for(int t = 1; t <= T+1; t++){
        if(t == T){
            data.close();
            data.open("../cifar.holdout");
            data << N << " " << D << " " << K << endl;
        } else if(t == T+1){
            data.close();
            data.open("../cifar.test");
            data << N << " " << D << " " << K << endl;
        }
        ifstream binary;
        if(t <= T){
            sprintf(filename,"data_batch_%d.bin",t);
        } else {
            sprintf(filename,"test_batch.bin");
        }
        cout << "Processing file " << filename << endl;
        binary.open(filename, ios::in | ios::binary);

        for (int i = 0; i < N; i++) {
            binary.read(label_buffer, 1);
            label = (int)label_buffer[0];
            if (label < 0 || label >= K) {
                cout << "Warning: image " << i << " has label " << label << endl;
                return 1;
            }
            binary.read(pixel_buffer, 3*NUM_PIXELS);
            for (int j = 0; j < D; j++) {
                data << extract_pixel(pixel_buffer, j, SCALE, USE_COLOR) << " ";
            }
            data << label;
            data << endl;
        }

        binary.close();
    }    
    data.close();
    return 0;
}
