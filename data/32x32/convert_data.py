import Image
import collections
import sys
import os

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
with open(dataset_name + '_ids.txt', 'w') as f:
    for i, c in enumerate(class_names):
        f.write('%d: %s\n' % (i, c))

data_ratios = {'holdout': 0.1, 'test': 0.1}
data_slices = collections.defaultdict(list)

for i, c in enumerate(class_names):
    data = [row + [i] for row in clusters[c]]
    n = len(data)

    for s in data_ratios:
        m = int(data_ratios[s] * n)
        data_slices[s].extend(data[:m])
        data = data[m:]

    data_slices['pretraining'].extend(data)


for s in data_slices:
    print s, len(data_slices[s])


def print_to_file(slice_name, data_slice):
    file_name = dataset_name + '_' + slice_name + '.txt'

    N = len(data_slice)
    K = len(clusters)
    D = row_len

    with open(file_name, 'w') as f:
        f.write('%d %d %d\n' % (N, D, K))
        for row in data_slice:
            f.write(' '.join(str(x) for x in row) + '\n')

for s in data_slices:
    print_to_file(s, data_slices[s])

