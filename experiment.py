import numpy
import numpy.linalg as linalg

filename = 'movement_libras.data'
#filename = 'testdata'

num_clusters = 0
clusters = {}
label_map = {}

for point in open(filename).readlines():
    point = point.strip().split(',')
    label = int(point[-1])
    if label not in label_map:
        label_map[label] = num_clusters
        clusters[num_clusters] = []
        num_clusters += 1
    clusters[label_map[label]].append(numpy.array([float(x) for x in point[:-1]]))

n = sum(len(cluster) for cluster in clusters.values())
k = len(clusters)
d = len(clusters.values()[0][0])
print '(n, k, d) = (%s, %s, %s)' % (n, k, d)

def mean(cluster):
    return sum(cluster)/len(cluster)

def covar(cluster):
    ExTx = sum(numpy.outer(x, x) for x in cluster)/len(cluster)
    mu = mean(cluster)
    return ExTx - numpy.outer(mu, mu) 

def logdet(A):
    (sign, val) = linalg.slogdet(A)
    return sign*val

mus = {label:mean(cluster) for (label, cluster) in clusters.items()}
sigmas = {label:covar(cluster) for (label, cluster) in clusters.items()}
invsigmas = {label:linalg.pinv(sigma) for (label, sigma) in sigmas.items()}
logdets = {label:logdet(sigma) for (label, sigma) in sigmas.items()}

def trace_prod(A, B):
    (d1, d2) = A.shape
    (d3, d4) = B.shape
    assert (d1 == d2 and d2 == d3 and d3 == d4)
    return sum(numpy.inner(A[i,...], B[...,i]) for i in range(len(A)))

subk = 4
M = numpy.zeros((subk, subk))

for i in range(subk):
    for j in range(subk):
        M[i,j] = (trace_prod(invsigmas[j], sigmas[i])
                  + numpy.inner(mus[i] - mus[j], numpy.dot(invsigmas[j], mus[i] - mus[j]))
                  + logdets[j])/-2.0

print M
