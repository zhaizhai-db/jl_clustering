import traceback
import sys
import collections

N, D, K = [int(x) for x in raw_input().strip().split()]
print '%d %d %d' % (N, D, K)
num_to_label = 5

label_counts = collections.defaultdict(int)
for i in xrange(N):
    row = [int(x) for x in raw_input().strip().split()]
    label = row[-1]
    label_counts[label] += 1
    if label_counts[label] > num_to_label:
        row[-1] = -1
    print ' '.join(str(x) for x in row)
