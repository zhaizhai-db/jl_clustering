import traceback
import sys
import collections

assert len(sys.argv) == 3, 'Usage: python remove_training_labels.py filename num_to_label'
filename = sys.argv[1]
num_to_label = int(sys.argv[2])

fin = open(filename + '.pretrain', 'r')
fout = open(filename + '.train', 'w')

N, D, K = [int(x) for x in fin.readline().strip().split()]
fout.write('%d %d %d' % (N, D, K))
fout.write('\n')

label_counts = collections.defaultdict(int)
for i in xrange(N):
    row = [int(x) for x in fin.readline().strip().split()]
    label = row[-1]
    label_counts[label] += 1
    if label_counts[label] > num_to_label:
        row[-1] = -1
    fout.write(' '.join(str(x) for x in row))
    fout.write('\n')
