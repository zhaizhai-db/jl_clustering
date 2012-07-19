import traceback
import sys
import collections

infilename = 'pretraining.txt'
outfilename = 'training.txt'

try:
    infile = open(infilename)
except Exception:
    print 'Expecting %s as input (couldn\'t open)' % infilename
    exit(0)

try:
    outfile = open(outfilename, 'w')
except Exception:
    print 'Couldn\'t open %s for writing' % outfilename
    exit(0)


N, D, K = [int(x) for x in infile.readline().strip().split()]
outfile.write('%d %d %d\n' % (N, D, K))
num_to_label = 5

label_counts = collections.defaultdict(int)
for i in xrange(N):
    row = [int(x) for x in infile.readline().strip().split()]
    label = row[-1]
    label_counts[label] += 1
    if label_counts[label] > num_to_label:
        row[-1] = -1
    outfile.write(' '.join(str(x) for x in row) + '\n')
