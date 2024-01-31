from CompressedGraph import *
from numpy import Inf
import numpy as np
from Potential import TablePotential
import time


p1 = TablePotential({
    (True, True): 4,
    (True, False): 1,
    (False, True): 1,
    (False, False): 3
}, symmetric=True)

d = Domain([1, 2, 3, 4])

time_start = time.perf_counter()

rv = []
for i in range(3):
    rv.append(RV(d))

f = []
for i in range(2):
    f.append(F(p1, (rv[i], rv[i+1])))

g = Graph()
g.rvs = rv
g.factors = f
g.init_nb()

cg = CompressedGraph(g)
cg.init_cluster()

for i in range(3):
    cg.split_factors()
    cg.split_rvs()

print(f'elapsed time {time.perf_counter() - time_start}')


