from Potential import GaussianPotential, MixturePotential, LaplacianPotential
from Potential import *
from HybridLBP import *
from numpy import Inf
from PBP import PBP
from EPBPLogVersion import EPBP
import matplotlib.pyplot as plt


def run_demo(method):
    domain = Domain((-500, 500), continuous=True)
    # domain = Domain((-5000, 5000), continuous=True)
    n = 10

    rv = []
    for _ in range(n):
        rv.append(RV(domain))

    # p1 = GaussianPotential([1, 2], [[2.0, 0.3], [0.3, 0.5]])
    p1 = GaussianPotential([100, 200], [[2.0, 0.3], [0.3, 0.5]])
    print(p1.inv)
    print(p1.coefficient)
    # p1 = LaplacianPotential()
    f = []
    for i in range(n - 1):
        f.append(F(p1, (rv[i], rv[i+1])))

    g = Graph()
    g.rvs = rv
    g.factors = f
    g.init_nb()

    # ground_truth = np.array([1.0, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 2.0])
    ground_truth = np.array([100, 239.91, 239.91, 239.91, 239.91, 239.91, 239.91, 239.91, 239.91, 200])
    mse = [] 
    mse_std = []
    for samples in range(10, 160, 10):  
        mse_per_sample_size = []
        for _ in range(10):
            if (method == 'EPBP'):
                bp = EPBP(g, n=samples, proposal_approximation='simple')
            elif (method == 'PBP'):
                bp = PBP(g, n=samples)
            else:
                bp = HybridLBP(g, samples)
            bp.run(10)
            # bp = PBP(g, n=50)
            # bp.run(10)
            bp_mode = []
            for x in rv:
                # print(bp.belief(1, x))
                # print(bp.map(x))
                bp_mode.append(bp.map(x))
            mode_value = np.array(bp_mode).flatten()
            print("samples size {} calculate mode value {}".format(samples, mode_value))
            mse_per_sample_size.append(np.mean((mode_value - ground_truth) ** 2))
        mse_per_sample_size = np.array(mse_per_sample_size)
        mse.append(np.mean(mse_per_sample_size))
        mse_std.append(np.std(mse_per_sample_size))          
    return [mse, mse_std]


def run_grid_demo(method):
    domain = Domain((-5, 15), continuous=True)
    # domain = Domain((-5000, 5000), continuous=True)
    n = 3

    rv = []
    for _ in range(n*n):
        rv.append(RV(domain))

    p1 = MixturePotential()
    p2 = LaplacianPotential()
    f = []
    for i in range(n - 1):
        for j in range(n):
            f.append(F(p2, (rv[i*n+j], rv[(i+1)*n+j])))

    for i in range(n):
        for j in range(n - 1):
            f.append(F(p2, (rv[i*n+j], rv[i*n+j+1])))

    for i in range(n):
        for j in range(n):
            f.append(F(p1, (rv[i*n+j],)))


    g = Graph()
    g.rvs = rv
    g.factors = f
    g.init_nb()

    # ground_truth = np.array([1.0, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 2.0])
    gt_grid = np.linspace(-5, 15, 200)
    probs = []
    probs1 = []
    samples = 5
    if (method == 'EPBP'):
        bp = EPBP(g, n=samples, proposal_approximation='simple')
    elif (method == 'PBP'):
        bp = PBP(g, n=samples)
    else:
        bp = HybridLBP(g, samples)
    bp.run(20)
    for val in gt_grid:
        if (method == 'PBP'):
            probs.append(bp.belief(val, rv[0])[0])
            probs1.append(bp.belief(val, rv[1*n+1])[0])
        elif (method == 'EPBP'):
            probs.append(bp.belief(val, rv[0]))
            probs1.append(bp.belief(val, rv[1*n+1]))
        else:
            pass         
    
    return [probs, probs1]

 
    # for samples in range(10, 20, 10):  
    #     mse_per_sample_size = []
    #     for _ in range(10):
    #         if (method == 'EPBP'):
    #             bp = EPBP(g, n=samples, proposal_approximation='simple')
    #         elif (method == 'PBP'):
    #             bp = PBP(g, n=samples)
    #         else:
    #             bp = HybridLBP(g, samples)
    #         bp.run(10)
    #         # bp = PBP(g, n=50)
    #         # bp.run(10)
    #         bp_mode = []
    #         for x in rv:
    #             # print(bp.belief(1, x))
    #             # print(bp.map(x))
    #             bp_mode.append(bp.map(x))
    #         mode_value = np.array(bp_mode).flatten()
    #         print("samples size {} calculate mode value {}".format(samples, mode_value))
    #         mse_per_sample_size.append(np.mean((mode_value - ground_truth) ** 2))
    #     mse_per_sample_size = np.array(mse_per_sample_size)
    #     mse.append(np.mean(mse_per_sample_size))
    #     mse_std.append(np.std(mse_per_sample_size))          
    return [mse, mse_std]


if __name__ == '__main__':
    # rls = []
    # for _ in range(10):
    #     rls.append(run_demo('PBP'))
    # rls = np.array(rls)
    # mean_rls = np.mean(rls, axis=0)-
    # std_rls = np.std(rls, axis=0)
    # print(mean_rls)
    # print(std_rls)
    # rlt = run_demo('PBP')
    # print("mse")
    # print(rlt[0])
    # print("mse_std")
    # print(rlt[1])

    rlt = run_grid_demo('EPBP')
    print("prob")
    print(rlt[0])
    print("prob1")
    print(rlt[1])
# PBP
# mse
# [0.03157918097656269, 0.01180550871093777, 0.010507673476562776, 0.012695141093750298, 0.011069821835937833, 0.010768335156250336, 0.01332450585937533, 0.011930615625000321, 0.011332507460937833, 0.010775720195312807, 0.010873151875000339, 0.010531576484375327, 0.01075538394531282, 0.010460519257812821, 0.01019880960937532]
# mse_std
# [0.021442740254513006, 0.004400824581575984, 0.003119181911135783, 0.0030887455209939554, 0.0026922440354273567, 0.0018687967664222148, 0.004204849747399153, 0.0016354422589736997, 0.001974976748317036, 0.002168631705624838, 0.0025849796794338827, 0.0015576214196820727, 0.00195174229334051, 0.0016098326526919064, 0.0016728253254476536]

# EPBP
# mse
# [0.02046931655901987, 0.014509587956539161, 0.013164941345608337, 0.012666857873597157, 0.013918097184971733, 0.012099269444874624, 0.011940791712468674, 0.010797912332730844, 0.012125043099946973, 0.010758964475828256, 0.011262656004505833, 0.011669017844770427, 0.010887464417506858, 0.010601100264124693, 0.011697901204312373]
# mse_std
# [0.008652821507675244, 0.004685459338528537, 0.0030848634316804754, 0.0018595762098662337, 0.004748135693179133, 0.0018993262226692973, 0.002237509604553494, 0.001981147235115501, 0.0022445678996760774, 0.0014359926151251191, 0.001009968886900661, 0.0014875422228832638, 0.0007057945994787471, 0.0012842289304947423, 0.0010535286763943246]



# PBP on shifting
# [5104659.80552, 5104659.80552, 5104659.80552, 5104659.80552, 5104659.80552, 5104659.80552, 5104659.80552, 5104659.80552, 5104659.80552, 5104659.80552, 5104659.80552, 5104659.80552, 5104659.80552, 5104659.80552, 5104659.80552]
# mse_std
# [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


# PBP on shifting
# mse
# [51045.44648, 51045.44648, 51045.44648, 51045.44648, 51045.44648, 51045.44648, 51045.44648, 51045.44648, 51045.44648, 51045.44648, 51045.44648, 51045.44648, 51045.44648, 51045.44648, 51045.44648]
# mse_std
# [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# EPBP on shifting
# mse
# [52439.22654852255, 51525.285073606836, 47826.48509259198, 50813.61012787907, 42554.92944764903, 49616.27114082851, 46900.4138666639, 46550.69966283937, 48422.2412800519, 44767.3163197655, 42929.23846878716, 47913.25566372282, 47690.172401810654, 46492.23550449848, 48240.83463513047]
# mse_std
# [9970.748574210054, 13597.017980295392, 7474.794596492886, 9126.24345317068, 7504.370678037581, 7405.622079757813, 9588.569951309479, 7785.9238577874185, 6703.309538850997, 8170.724761152645, 8040.057253510799, 8170.7496805543715, 7939.035357807124, 6348.6444559451875, 7449.591457926]
