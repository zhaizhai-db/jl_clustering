from evaluate import *

N = 360
K = 15

correct_assign = [int(l.strip().split(',')[-1]) for l in open('movement_libras.data').readlines()]


def read_assignments(path):
    assignments = []
    infile = open(path)
    for l in infile:
        a = eval(l.strip())
        if a:
            assignments.append(a)
    return assignments

em_assignments = read_assignments('em_assignments.txt')
jl_assignments = read_assignments('jl_assignments.txt')
print 'finished reading in files'

#assert len(em_assignments) == 100
for a in em_assignments + jl_assignments:
    assert len(a) == 360


measure = wallace

def similarity(l1, l2):
    count = 0
    avg = 0.0
    for a1 in l1:
        for a2 in l2:
            if a1 is not a2:
                count += 1
                avg += measure(a1, a2)
    return avg / count

em_assignments = em_assignments[:10]
jl_assignments = jl_assignments[:10]

print 'using', measure.__name__
print 'em-em', similarity(em_assignments, em_assignments)
print 'em-jl', similarity(em_assignments, jl_assignments)
print 'jl-jl', similarity(jl_assignments, jl_assignments)


