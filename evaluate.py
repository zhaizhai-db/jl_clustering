# Takes two partitions and returns the Rand index of similarity (0 <= x <= 1)
# Always returns 1 for equal clusters
def rand_index(part1, part2):
    assert(len(part1) == len(part2))
    n = len(part1)
    matches = 0
    for i in range(n):
        for j in range(n):
            if i != j and ((part1[i] == part1[j]) == (part2[i] == part2[j])):
                matches += 1
    return 1.0*matches/(n*(n - 1))

# Takes two partitions and returns the correct Rand index of similarity (x <= 1)
# Always returns 1 on equal partitions
def wallace(part1, part2):
    assert(len(part1) == len(part2))
    n = len(part1)
    num_joined_both = 0
    num_joined_either = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                if (part1[i] == part1[j]) and (part2[i] == part2[j]):
                    num_joined_both += 1
                if (part1[i] == part1[j]) or (part2[i] == part2[j]):
                    num_joined_either += 1
    if num_joined_either == 0:
        return 1
    return 1.0*num_joined_both/num_joined_either

# Takes two partitions and returns the correct Rand index of similarity (x <= 1)
# Always returns 1 on equal data
def corrected_rand(part1, part2):
    assert(len(part1) == len(part2))
    n = len(part1)
    k = max(len(set(part1)), len(set(part2)))
    matches = 0
    for i in range(n):
        for j in range(n):
            if i != j and ((part1[i] == part1[j]) == (part2[i] == part2[j])):
                matches += 1
    if k == 1:
        return 1
    return (matches - 1.0*(k*k - 2*k + 2)/(k*k)*n*(n - 1)) \
            / (2.0*(k-1)/(k*k)*n*(n - 1))

# Takes two partitions and returns the corrected Wallace index of similarity (x <= 1)
# Does NOT always return 1 on equal data
def corrected_wallace(part1, part2):
    assert(len(part1) == len(part2))
    n = len(part1)
    k = max(len(set(part1)), len(set(part2)))
    num_joined_both = 0
    for i in range(n):
        for j in range(n):
            if i != j and (part1[i] == part1[j]) and (part2[i] == part2[j]):
                num_joined_both += 1
    if k == 1:
        return 1
    return (num_joined_both - 1.0/(k*k)*n*(n - 1))/(n*(n - 1) - 1.0/(k*k)*n*(n - 1))
