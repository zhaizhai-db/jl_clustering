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
k0 = 0.1
v = 3.0
print '(n, k, d) = (%s, %s, %s)' % (n, k, d)

def mean(cluster):
    return sum(cluster)/len(cluster)

def covar(cluster):
    ExTx = sum(numpy.outer(x, x) for x in cluster)/len(cluster)
    mu = mean(cluster)
    return ExTx - numpy.outer(mu, mu) 

def mu(cluster):
    global k0
    l = len(cluster)
    return k0*sum(cluster)/(k0+l)

def sigma(cluster):
    global d, k0, v
    l = len(cluster)
    xbar = mean(cluster)
    cov = covar(cluster)
    Tn = numpy.identity(d) + l*cov + k0*l/(k0+l) * numpy.outer(xbar,xbar)
    return Tn*(k0+l+1)/((k0+l)*(v+l))

def logdet(A):
    (sign, val) = linalg.slogdet(A)
    return sign*val

mus = dict((label,mu(cluster)) for (label, cluster) in clusters.items())
sigmas = dict((label,sigma(cluster)) for (label, cluster) in clusters.items())
invsigmas = dict((label,linalg.pinv(sigma)) for (label, sigma) in sigmas.items())
logdets = dict((label,logdet(sigma)) for (label, sigma) in sigmas.items())

def trace_prod(A, B):
    (d1, d2) = A.shape
    (d3, d4) = B.shape
    assert (d1 == d2 and d2 == d3 and d3 == d4)
    return sum(numpy.inner(A[i,...], B[...,i]) for i in range(len(A)))

def estimate_logpdf(i, j):
    global d, v
    li = len(clusters[i])
    lj = len(clusters[j])
    mu1 = mus[i]
    mu2 = mus[j]
    Sigma1 = sigmas[i]
    num_trials = 1000
    # first, sample from a multivariate student-t with parameters mu1, Sigma1
    y = numpy.random.multivariate_normal(0.0*mu1, Sigma1, size=num_trials)
    u = numpy.random.chisquare(v+li, size=(num_trials,1))
    x = y * numpy.sqrt((v+li)/numpy.tile(u,(1,d))) + numpy.tile(mu1,(num_trials,1))
    x2 = x - numpy.tile(mu2,(num_trials,1))
    r = numpy.sum(numpy.dot(x2, invsigmas[j]) * x2, 1)
    return numpy.mean(r)
    #return -0.5*(logdets[j]+(v+lj+d)*numpy.mean(numpy.log(1+r/(v+lj))))



subk = k
M = numpy.zeros((subk, subk))

for i in range(subk):
    for j in range(subk):
        M[i,j] = estimate_logpdf(i,j)
                 #(trace_prod(invsigmas[j], sigmas[i])
                 # + numpy.inner(mus[i] - mus[j], numpy.dot(invsigmas[j], mus[i] - mus[j]))
                 # + logdets[j])/-2.0

print M
