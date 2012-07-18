import numpy

clusters = {}
for point in open('movement_libras.data').readlines():
    point = point.strip().split(',')
    label = int(point[-1])
    if label not in clusters:
        clusters[label] = []
    clusters[label].append([float(x) for x in point[:-1]])

n = sum(len(cluster) for cluster in clusters.values())
k = len(clusters)
d = len(clusters.values()[0][0])
print '(n, k, d) = (%s, %s, %s)' % (n, k, d)


