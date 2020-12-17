#!/usr/bin/env python
# coding: utf-8

import networkx as nx
import numpy as np
import gc
import scipy


G = nx.read_edgelist("com-amazon.ungraph.txt", create_using=nx.Graph, nodetype = int)
n = G.number_of_nodes() # use this but in a ordered way
print(list(G.nodes)[:10])

nodes = sorted(list(G.nodes))
# n = max(list(G.nodes))
# print("n: ", n)
# print("m: ", m)
# exit()
L = nx.normalized_laplacian_matrix(G,  nodelist=nodes)

A = nx.adjacency_matrix(G, nodelist=nodes) # zero based
# to do create a node id -> row id, dictionary
row2id = dict( enumerate(nodes))  # zero-based
id2row =  dict( (node_id, row) for (row, node_id) in row2id.items())



# new
D_diags = np.sum(A, axis = 1)
D_diags = D_diags.squeeze(1)
D_diags = D_diags.tolist()[0]
D = scipy.sparse.diags(D_diags)


# # save memory
# del G
#
# gc.collect()


I = scipy.sparse.diags([1]*n)

one_vector = scipy.sparse.bsr_matrix([1]*n).transpose()

v_1 = np.sqrt(D).dot(one_vector).todense()



from scipy import stats
x = scipy.stats.multivariate_normal.rvs(mean=0, cov=1, size=(n,1))
x = scipy.sparse.csr_matrix(x)


v_1_norm = np.linalg.norm(v_1, 2)
y = x.transpose() - (x.dot(v_1*(1/v_1_norm)))[0,0]*(v_1*(1/v_1_norm))
# print(y.shape)
# print(x.dot(v_1*(1/v_1_norm))[0,0])
# print((v_1*(1/v_1_norm)).shape)

def power_method(y, M, k = 1000):
    count = 0
    while count < k:
        y = y*(1/np.linalg.norm(y,2))
        prev_y = y
        y = M*y


        update =  np.linalg.norm(y-prev_y, 2)
        if count % 10 == 0:
            print("update: ", update)
        count += 1
    eigenval = (y.transpose()@M@y)/(y.transpose()@y)
    return y, eigenval
M = (2*I-L)
v_2, eigenval = power_method(y, M)

D_csc = scipy.sparse.csc_matrix(D) # it's more efficient for spsolve



scipy.sparse.diags(1/np.sqrt(D_csc.diagonal()))


# In[75]:


from scipy.sparse.linalg import spsolve


vertices = scipy.sparse.diags(1/np.sqrt(D_csc.diagonal())) * v_2


vertices_l = vertices.flatten().tolist()[0]

indexed_vertices_l = list(enumerate(vertices_l))  # row index and value

sorted_indexed_vertices_l = sorted(indexed_vertices_l, key = lambda t:t[1])


def degree(i):
    return D.data[0][i]
D.data[0][298945]

vol_V = np.sum(D.data)

all_nodes_S = set(list(range(n)))
S = set()
# set(G.nodes)
conductance = 0
vol_S = 0
min_conductance = 99
cut = 0
for indexed_vertex in sorted_indexed_vertices_l:
    v_row_idx, v = indexed_vertex

    num_intra = 0
    num_inter = 0
    for neighbour in A[v_row_idx].indices: # loop over the neighbours and check if it's in S
        # neighbour is node id
        if neighbour in S: 
            num_intra += 1
        else:
            num_inter += 1
    cut = cut - num_intra + num_inter
    vol_S = vol_S + degree(v_row_idx)
    conductance = cut / min(vol_S, vol_V-vol_S)
    S.add(v_row_idx)
    if conductance < min_conductance:
        min_conductance = conductance
        min_set = S if vol_S <= (vol_V-vol_S) else all_nodes_S - S
        print("min_conductance: ", min_conductance)

print("final min_conductance: ", min_conductance)
print("the node ids of min set is: ", [row2id[row_idx] for row_idx in min_set])


# no, keep the vertices matrix

# extract the diagonal and label them
# sorted( , lambda )
# sweep
# |E| - number of S intra edge + number of (S,T) inter edges
# |vol| + degree

