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
#define SCALE (2)
#define USE_COLOR (false)

// N can be any positive integer up to (and including) 10000.
// The training set will have (T-1)*N points, while the holdout
// and test data will have N points each.
#define N (1000)
#define D ((USE_COLOR ? 3 : 1)*NUM_PIXELS/(SCALE*SCALE))
#define T (5)

// This dataset has a coarse and fine label. There are 20 coarse
// categories and 100 fine categories.
#define USE_FINE_LABELING (true)
#define K ((USE_FINE_LABELING ? 100 : 20))

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
    char label_buffer[2];
    char pixel_buffer[3*NUM_PIXELS];

    ofstream data;
    data.open("../bigcifar.pretrain");
    data << ((T-1)*N) << " " << D << " " << K << endl;

    char filename[1000];
    ifstream binary;
    sprintf(filename,"train.bin");
    cout << "Processing file " << filename << endl;
    binary.open(filename, ios::in | ios::binary);

    for(int t = 1; t <= T+1; t++){
        if(t == T){
            data.close();
            data.open("../bigcifar.holdout");
            data << N << " " << D << " " << K << endl;
        } else if(t == T+1){
            data.close();
            data.open("../bigcifar.test");
            data << N << " " << D << " " << K << endl;

            binary.close();
            sprintf(filename,"test.bin",t);
            cout << "Processing file " << filename << endl;
            binary.open(filename, ios::in | ios::binary);
        }

        for (int i = 0; i < N; i++) {
            binary.read(label_buffer, 2);
            label = (int)label_buffer[(USE_FINE_LABELING ? 1 : 0)];
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
    }    
    data.close();
    return 0;
}
