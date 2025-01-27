#!/usr/bin/env python3
# This file is based from https://github.com/jeremyzhangsq/AESC
# It is solely used for regularizing input files across the implementations
# Please refer the original author for the license
# This will be replace by a C++ implementation
import argparse
import time
import sys

import scipy.sparse as sp
import networkx as nx
import numpy as np
from sklearn import preprocessing
from scipy.sparse import csr_matrix
from os.path import join 
from networkx.utils import reverse_cuthill_mckee_ordering, cuthill_mckee_ordering
from networkx.relabel import relabel_nodes
from pathlib import Path

G = None
m = None
n = None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process...')
    parser.add_argument('-f', type=str, default="facebook/", help='dataset folder')
    parser.add_argument('-w', type=int, default=128, help='omega')
    args = parser.parse_args()
    fname = args.f
    print(args)
    ifile = join(fname, 'graph.txt')

    G = nx.read_edgelist(ifile, nodetype=int, create_using=nx.Graph)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    print(n, m)
    # ifile = '{}/graph.txt'.format(fname)
    ifile = join(fname, 'graph.txt')
    print("f=%s, n=%d, m=%d, ad=%f"%(fname,n,m,2.0*m/n))
    A = csr_matrix(nx.adjacency_matrix(G, nodelist=sorted(G.nodes()),dtype=np.single))
    P = preprocessing.normalize(A, norm='l1', axis=1)
    d = np.array(A.sum(axis=1).squeeze()).reshape(-1)*0.5/m
    D = sp.diags(d, shape=(n,n))
    Dsqrt = np.sqrt(D)
    a = [1.0]*n
    dinv = np.divide(a, d, out=np.zeros_like(a), where=d!=0)
    Dneg = sp.diags(dinv, shape=(n,n))
    Dnegsqrt = np.sqrt(Dneg)
    P = Dsqrt.dot(P.dot(Dnegsqrt))
    omega = args.w
    # ofile = '{}/sorted_eigens_{}.txt'.format(fname,omega)
    ofile = join(fname, "sorted_eigens_{}.txt".format(omega))
    print("HERE")
    del A
    start = time.time()
    vals, vecs = sp.linalg.eigsh(P, omega, which='LM')
    vals = vals.tolist()
    print("omega:{},time:{}".format(omega, time.time() - start))
    vecs = Dnegsqrt.dot(vecs)
    vecs = vecs.transpose().tolist()
    eigens = [each for each in zip(vals, vecs)]
    eigens = sorted(eigens, key=lambda x: abs(x[0]), reverse=True)  # sort by the absolute value of eigen values
    statfilename = join(fname, 'stat.txt')
    open(statfilename,'w').write("n=%d\nm=%d\nl=%f"%(n,m,eigens[0][0]))
    with open(ofile, "w") as f:
        for val, vec in eigens:
            # f.write(str(val)+"\n")
            vecstr = " ".join(str(e) for e in vec)
            f.write(str(val)+" "+vecstr+"\n")
