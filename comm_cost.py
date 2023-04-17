# This file counts the per-node communication cost for 2-layer MLP. 
#   The model contains two layers:
#       the first layer accepts input of size dims[0] * dims[1] 
#       and multiplies it by a weight matrix of size dims[1] * dims[2], 
# 
#       the second layer accepts input of size dims[0] * dims[2]
#       and multiplies it by a weight matrix of size dims[2] * dims[3]
#
#   dims: a list of length 3, containing [B, F, H]
#   partition1: partition of the first layer, containing [nb, nf1, nf2, nh]
#   partition2: partition of the second layer, containing [nb, nh1, nh2, nf]

B = 100
F = 100
H = 4 * F

def calc_comm_cost_layer(dims, partition, verbose=False):
    b, f, h = dims
    nb, nf1, nf2, nh = partition
    cost = 0

    # 1) Forward: Allreduce cost before ReLu
    if nf1 > 1 or nf2 > 1:
        r = nf1 * nf2 * (b / nb) * (h / nh)
        cost += r
        if verbose:
            print("  - Forward: Allreduce cost before ReLu: %.2f" % r)
    
    # 2) Forward: dimension transition cost
    r = (b / nb) * (h / nh)
    cost += r
    if verbose:
        print("  - Forward: dimension transition cost: %.2f" % r)

    # 3) Backward: Allreduce cost for weight
    if nb > 1:
        r = nb * (f / nf2) * (h / nh)
        cost += r
        if verbose:
            print("  - Backward: Allreduce cost for weight: %.2f" % r)

    # 4) Backward: dimension transition cost
    r = (b / nb) * (f / nf1)
    cost += r
    if verbose:
        print("  - Backward: dimension transition cost: %.2f" % r)

    if verbose:
        print("  - Total per node communication cost this layer: %.2f" % cost)
    return cost


# calc_comm_cost(dims, partition1, partition2)
#   Calculates the total communication cost of the model
def calc_comm_cost(dims, partition1, partition2, verbose=False):
    if verbose:
        print("Calculating the communication cost for 2-layer MLP with:")
        print(' * Batch size (B):', dims[0])
        print(' * Input feature size (F):', dims[1])
        print(' * Hidden feature size (H):', dims[2])
        print(' * Partition of the first layer:', partition1)
        print(' * Partition of the second layer:', partition2)
        print()

    cost = 0

    # cost for first layer
    if verbose:
        print("Calculating the communication cost for the first layer:")
    r = calc_comm_cost_layer(dims, partition1, verbose)
    cost += r
    if verbose:
        print("Total per node communication cost for the first layer: %.2f" % r)
        print()

    # cost for second layer
    if verbose:
        print("Calculating the communication cost for the second layer:")
    r = calc_comm_cost_layer(dims, partition2, verbose)
    cost += r
    if verbose:
        print("Total per node communication cost for the second layer: %.2f" % r)
        print()

    if verbose:
        print("Total per node communication cost for the model: %.2f" % cost)
    return cost

def calc_num_nodes(dims, partition1, partition2, verbose=False):
    n_node1 = partition1[0] * partition1[1] * partition1[2] * partition1[3]
    n_node2 = partition2[0] * partition2[1] * partition2[2] * partition2[3]

    if verbose:
        print(" * number of nodes in the first layer: %d" % n_node1)
        print(" * number of nodes in the second layer: %d" % n_node2)
        print(" * total node number: %d" % (n_node1 + n_node2))
        print()
    
    return n_node1 + n_node2

def calc_mem_cost(dims, partition1, partition2, verbose=False):
    cost = 0
    cost += calc_mem_cost_layer(dims, partition1, verbose)
    cost += calc_mem_cost_layer(dims, partition2, verbose)
    if verbose:
        print("Total per node memory cost for the model: %.2f" % cost)
    return cost

def calc_mem_cost_layer(dims, partition, verbose=False):
    b, f, h = dims
    nb, nf1, nf2, nh = partition
    return (b / nb) * (f / nf1) + (f / nf2) * (h / nh) + 2 * (b / nb) * (h / nh)

def valid_partition(num_nodes):
    def factor(n):
        return [i for i in range(1, n + 1) if n % i == 0]

    def valid_partition_helper(n, factors):
        for i in factors:
            for j in factors:
                for k in factors:
                    for l in factors:
                        if i * j * k * l <= n:
                            yield [i, j, k, l]

    factors = factor(num_nodes)
    return ((a, b) for a in valid_partition_helper(num_nodes, factors) for b in valid_partition_helper(num_nodes, factors))

if __name__ == '__main__':
    dims = [B, F, H]
    # partition1 = [2, 2, 2, 2]
    # partition2 = [2, 2, 2, 2]
    # print(calc_comm_cost(dims, partition1, partition2, True))
    # print(calc_mem_cost(dims, partition1, partition2, True))

    results = []
    for p1, p2 in valid_partition(16):
        comm_cost = calc_comm_cost(dims, p1, p2, False)
        num_nodes = calc_num_nodes(dims, p1, p2, False)
        mem_cost = calc_mem_cost(dims, p1, p2, False)
        results.append(p1 + p2 + [comm_cost, num_nodes, mem_cost])

    print("B, F1, F2, H, B, H1, H2, F, comm_cost, num_nodes, mem_cost")
    for r in sorted(results, key=lambda x: x[8]):
        print(','.join(map(str, r)))

