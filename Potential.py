from Graph import Potential
import numpy as np
from math import pow, pi, e, sqrt, exp
from scipy.stats import norm, gumbel_r, laplace


class MixturePotential(Potential):
    def __init__(self):
        Potential.__init__(self, symmetric=False)
        self.mu_gaussian = np.array([-2.])
        self.sigma_gaussian = np.array([1.])
        self.mu_gumbel = np.array([2.])
        self.beta_gumbel = np.array([1.3])
        self.alpha_1 = 0.6
        self.alpha_2 = 0.4

    def get(self, particles):
        # gaussian_part = np.exp(-np.power(np.array(particles[0]) - self.mu_gaussian, 2)/(2*self.sigma_gaussian**2))
        gaussian_part = norm.pdf(np.array(particles[0]), loc=self.mu_gaussian, scale=self.sigma_gaussian)

        # gumbel_part = np.exp(-(np.array(particles[0]) - self.mu_gumbel)/self.beta_gumbel - np.exp(-(np.array(particles[0]) - self.mu_gumbel)/self.beta_gumbel))
        gumbel_part = gumbel_r.pdf(np.array(particles[0]), loc=self.mu_gumbel, scale=self.beta_gumbel)
        potential = self.alpha_1 * gaussian_part + self.alpha_2 * gumbel_part
        return potential
    
    def _soften(self, values):
        '''
        Soften the potential values to avoid numerical issues in the meantime maintain the probability property
        '''
        up_bd = np.ones_like(values)*0.9999
        low_bd = np.ones_like(values)*0.0001
        soften = np.maximum(np.minimum(values, up_bd), low_bd)
        normalized = soften/np.sum(soften)
        return normalized


class TablePotential(Potential):
    def __init__(self, table, symmetric=False):
        Potential.__init__(self, symmetric=symmetric)
        self.table = table

    def get(self, parameters):
        return self.table[parameters]


class LaplacianPotential(Potential):
    def __init__(self):
        Potential.__init__(self, symmetric=False)
        self.mu_laplace = np.array([0.])
        self.beta_laplace = np.array([2.])

    def get(self, particles):
        # potential = np.exp(-np.abs(np.array(particles[0]), np.array(particles[1]) - self.mu_laplace)/self.beta_laplace)'
        potential = laplace.pdf(np.array(particles[0]) - np.array(particles[1]), loc=self.mu_laplace, scale=self.beta_laplace)
        return potential

    def _soften(self, values):
        '''
        Soften the potential values to avoid numerical issues in the meantime maintain the probability property
        '''
        up_bd = np.ones_like(values)*0.9999
        low_bd = np.ones_like(values)*0.0001
        soften = np.maximum(np.minimum(values, up_bd), low_bd)
        normalized = soften/np.sum(soften)
        return normalized


class GaussianPotential(Potential):
    def __init__(self, mu, sig, w=1):
        Potential.__init__(self, symmetric=False)
        self.mu = np.array(mu)
        self.sig = np.matrix(sig)
        self.inv = self.sig.I
        det = np.linalg.det(self.sig)
        p = float(len(mu))
        if det == 0:
            raise NameError("The covariance matrix can't be singular")
        self.coefficient = w / (pow(2*pi, p*0.5) * pow(det, 0.5))

    def get(self, parameters):
        x_mu = np.matrix(np.array(parameters) - self.mu)
        return self.coefficient * pow(e, -0.5 * (x_mu * self.inv * x_mu.T))


class LinearGaussianPotential(Potential):
    def __init__(self, coeff, sig):
        Potential.__init__(self, symmetric=False)
        self.coeff = coeff
        self.sig = sig

    def get(self, parameters):
        return np.exp(-(parameters[1] - self.coeff * parameters[0]) ** 2 * 0.5 / self.sig)

    def __hash__(self):
        return hash((self.coeff, self.sig))

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__ and
            self.coeff == other.coeff and
            self.sig == other.sig
        )


class X2Potential(Potential):
    def __init__(self, coeff, sig):
        Potential.__init__(self, symmetric=False)
        self.coeff = coeff
        self.sig = sig

    def get(self, parameters):
        return np.exp(-self.coeff * parameters[0] ** 2 * 0.5 / self.sig)

    def __hash__(self):
        return hash((self.coeff, self.sig))

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__ and
            self.coeff == other.coeff and
            self.sig == other.sig
        )


class XYPotential(Potential):
    def __init__(self, coeff, sig):
        Potential.__init__(self, symmetric=True)
        self.coeff = coeff
        self.sig = sig

    def get(self, parameters):
        return np.exp(-self.coeff * parameters[0] * parameters[1] * 0.5 / self.sig)

    def __hash__(self):
        return hash((self.coeff, self.sig))

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__ and
            self.coeff == other.coeff and
            self.sig == other.sig
        )


class ImageNodePotential(Potential):
    def __init__(self, mu, sig):
        Potential.__init__(self, symmetric=True)
        self.mu = mu
        self.sig = sig

    def get(self, parameters):
        u = (parameters[0] - parameters[1] - self.mu) / self.sig
        return exp(-u * u * 0.5) / (2.506628274631 * self.sig)


class ImageEdgePotential(Potential):
    def __init__(self, distant_cof, scaling_cof, max_threshold):
        Potential.__init__(self, symmetric=True)
        self.distant_cof = distant_cof
        self.scaling_cof = scaling_cof
        self.max_threshold = max_threshold
        self.v = pow(e, -self.max_threshold / self.scaling_cof)

    def get(self, parameters):
        d = abs(parameters[0] - parameters[1])
        if d > self.max_threshold:
            return d * self.distant_cof + self.v
        else:
            return d * self.distant_cof + pow(e, -d / self.scaling_cof)
