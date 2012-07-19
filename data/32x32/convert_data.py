import Image
import collections
import sys
import os
from random import randint, sample

DATA_RATIOS = {'holdout': 0.33, 'test': 0.33}
DOWNSAMPLE_RATE_N = 0.4
DOWNSAMPLE_RATE_D = 0.1
DOWNSAMPLE = True

def dot(v, u):
    return sum(x*y for (x, y) in zip(v, u))

def mult(A, v):
    return [dot(u, v) for u in A]

# Returns a function defining a random projection from Rn to Rm
def binary_projection(n, m):
    proj = []
    for i in range(m):
        proj.append([2*randint(0, 1) - 1 for x in xrange(n)])
    return lambda v: mult(proj, v)

def downsample(clusters, rate_n, rate_d):
    new_clusters = {}
    d = len(clusters.values()[0][0])
    new_d = int(rate_d * d)
    proj = binary_projection(d, new_d) if new_d < d else lambda x: x
    for c in clusters:
        new_n = int(rate_n * len(clusters[c]))
        new_clusters[c] = [proj(x) for x in sample(clusters[c], new_n)]
    return new_clusters

dataset_name = '32x32'
clusters = collections.defaultdict(list)
row_len = None

img_list = open('imgs.txt')
for line in img_list:
    l = line.strip()
    if not l:
        continue

    c = l.split('_')[0].lower()
    img_path = os.path.join('jpgs', l)
    row = list(Image.open(img_path).getdata())
    clusters[c].append(row)

    if row_len is None:
        row_len = len(row)
    assert row_len == len(row)

class_names = list(clusters)
f = open(dataset_name + '_ids.txt', 'w')
for i, c in enumerate(class_names):
    f.write('%d: %s\n' % (i, c))
f.close()

if DOWNSAMPLE:
    print 'Downsampling...'
    clusters = downsample(clusters, DOWNSAMPLE_RATE_N, DOWNSAMPLE_RATE_D)
    print 'Done downsampling'

data_slices = collections.defaultdict(list)

for i, c in enumerate(class_names):
    data = [row + [i] for row in clusters[c]]
    n = len(data)

    for s in DATA_RATIOS:
        m = int(DATA_RATIOS[s] * n)
        data_slices[s].extend(data[:m])
        data = data[m:]

    data_slices['pretrain'].extend(data)


for s in data_slices:
    print s, len(data_slices[s])


def print_to_file(slice_name, data_slice):
    file_name = dataset_name + '.' + slice_name

    N = len(data_slice)
    K = len(clusters)
    D = len(clusters.values()[0][0])

    f = open('../' + file_name, 'w')
    f.write('%d %d %d\n' % (N, D, K))
    for row in data_slice:
        f.write(' '.join(str(x) for x in row) + '\n')
    f.close()

for s in data_slices:
    print_to_file(s, data_slices[s])

