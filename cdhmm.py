import numpy as np
from scipy.stats import norm
import scipy.stats as stats
import seaborn as sns
import math

class simple_cdhmm(object):
    def __init__(self, x, means, variances, pi, A, exit_p):
        self.x = x
        self.means = means
        self.variances = variances
        self.pi = pi
        self.A = A
        self.exit_p = exit_p

    def get_probability_densities(self):
        Em = []
        for m, v in zip(self.means, self.variances):
            Em.append(stats.norm.pdf(self.x, m, math.sqrt(v)))
        Em = np.asarray(Em)
        self.Em = Em
        return Em

    def forward(self):
        T = self.Em.shape[1]
        N = self.A.shape[0]
        alpha = np.zeros((T, N))
        alpha[0] = self.pi*self.Em[:,0]
        for t in range(1, T):
            alpha[t][0] = np.inner(alpha[t-1], self.A[:,0]) * self.Em[0, t]
            alpha[t][1] = np.inner(alpha[t-1], self.A[:,1]) * self.Em[1, t]
        self.alpha = alpha
        return alpha

    def likelihood(self):
        l = self.alpha.shape[0]
        self.forward_likelihood = np.inner(self.alpha[l-1], self.exit_p)
        return  self.forward_likelihood


    def backward(self):
        N = self.A.shape[0]
        T = self.Em.shape[1]
        beta = np.zeros((T,N))
        beta[T-1] = self.exit_p        
        for t in reversed(range(T-1)):
            y = 0
            for n in range(N):
                y = y + (beta[t+1,n] * self.A[0,n] * self.Em[n, t+1])
            beta[t][0] = y
            z = 0
            for n in range(N):
                z = z + (beta[t+1,n] * self.A[1,n] * self.Em[n, t+1])
            beta[t][1] = z
        self.beta = beta
        return beta

    def likelihood_b(self):
        sum = 0
        for n in range (self.beta.shape[1]):
            sum = sum + self.pi[n]*self.Em[n, 0]*self.beta[0, n]
        self.backward_likelihood = sum
        return sum

    def get_occupation_likelihoods(self):
        occ = np.zeros((self.beta.shape[0], self.beta.shape[1]))
        for t in range(self.beta.shape[0]):
            for i in range (self.beta.shape[1]):
                occ[t][i] = (self.alpha[t, i]*self.beta[t, i])/self.forward_likelihood
        self.occ = occ
        return occ

    def getNewMean(self):
        val = np.zeros(self.occ.shape[0])
        new_means = []
        for i in range(self.occ.shape[1]):    
            for j in range(self.occ.shape[0]):
                val[j] = self.occ[j][i]*self.x[j]
            new_means.append(np.sum(val)/np.sum(self.occ[:, i]))
        self.new_means = new_means
        return new_means
    

    def getNewVariance(self):
        val = np.zeros(self.occ.shape[0])
        new_vars = []
        for i in range(self.occ.shape[1]):
            k = 0

            for j in range(self.occ.shape[0]):
                k+= 1
                val[j] = self.occ[j][i]*(self.x[j] - self.means[i])*(self.x[j] - self.means[i])

            new_vars.append(np.sum(val)/np.sum(self.occ[:, i]))
        self.new_vars = new_vars
        return new_vars
