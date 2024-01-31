import pandas as pd
from Potential import *
from Graph import *
from HybridLBPLogVersion import HybridLBP
from EPBPLogVersion import EPBP
from GaLBP import GaLBP
from GaBP import GaBP
from PBP import PBP
import numpy as np

import time

def run_demo(method):
    p1 = MixturePotential()
    p2 = LaplacianPotential()

    domain = Domain((-10, 10), continuous=True)

    row = 3
    col = 3

    rvs = []
    for _ in range(row * col):
        rvs.append(RV(domain))

    fs = []

    # create hidden-obs factors
    pxo = MixturePotential()
    for i in range(row):
        for j in range(col):
            fs.append(
                F(
                    pxo,
                    (rvs[i * col + j],)
                )
            )

    # create hidden-hidden factors
    pxy = LaplacianPotential()
    for i in range(row):
        for j in range(col - 1):
            fs.append(
                F(
                    pxy,
                    (rvs[i * col + j], rvs[i * col + j + 1])
                )
            )
    for i in range(row - 1):
        for j in range(col):
            fs.append(
                F(
                    pxy,
                    (rvs[i * col + j], rvs[(i + 1) * col + j])
                )
            )

    g = Graph()
    g.rvs = rvs 
    g.factors = fs
    g.init_nb()

    rlt = []
    for samples in range(1, 10, 1):
        # bp = HybridLBP(g, n=10, proposal_approximation='simple')
        # bp.run(10, c2f=0, log_enable=False)
        # bp = EPBP(g, n=50, proposal_approximation='simple')
        if (method == 'EPBP'):
            bp = EPBP(g, n=samples, proposal_approximation='simple')
        elif (method == 'PBP'):
            bp = PBP(g, n=samples)
        else:
            bp = HybridLBP(g, samples)
        bp.run(10)
        # bp = GaLBP(g)
        # bp.run(20, log_enable=False)
        # def initial_proposal():
        #     for i in range(row):
        #         for j in range(col):
        #             bp.q[rvs[i * col + j]] = (m[i, j], 2)
        #
        # bp.custom_initial_proposal = initial_proposal

        # reconstruct image
        m_hat = np.zeros((row, col))
        for i in range(row):
            for j in range(col):
                m_hat[i, j] = bp.map(rvs[i * col + j])

        print(m_hat)
        mode_value = np.array(m_hat).flatten()
        rlt.append(mode_value)
    return rlt

if __name__ == '__main__':
    # rls = []
    # for _ in range(10):
    #     rls.append(run_demo('PBP'))
    # rls = np.array(rls)
    # mean_rls = np.mean(rls, axis=0)
    # std_rls = np.std(rls, axis=0)
    # print(mean_rls)
    # print(std_rls)
    rlt = run_demo('PBP')
    print(rlt)