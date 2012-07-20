import Image
import collections
import sys
import os
from random import randint, sample, shuffle

DOWNSAMPLE_RATE_N = 1.0 #0.4
DOWNSAMPLE_RATE_D = 1.0 #0.25
DOWNSAMPLE = False

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

for data_type in ('pretrain', 'holdout', 'test'):
    dataset_name = '32x32'
    clusters = collections.defaultdict(list)
    row_len = None

    img_list = open('%s.txt' % (data_type,))
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

    if DOWNSAMPLE:
        print 'Downsampling %s...' % (data_type,)
        clusters = downsample(clusters, DOWNSAMPLE_RATE_N, DOWNSAMPLE_RATE_D)

    class_names = list(clusters)
    f = open(dataset_name + 'ids.txt', 'w')
    for i, c in enumerate(class_names):
        clusters[c] = [x + [i] for x in clusters[c]]
        f.write('%d: %s\n' % (i, c))
    f.close()

    print 'Dumping %s to file...' % (data_type,)
    data = sum(clusters.values(), [])
    shuffle(data)
    N = len(data)
    D = len(data[0])
    K = len(clusters)

    file_name = dataset_name + '.' + data_type
    f = open('../' + file_name, 'w')
    f.write('%d %d %d\n' % (N, D, K))
    for row in data:
        f.write(' '.join(str(x) for x in row) + '\n')
    f.close()
    print 'Done with %s.' % (data_type,)
