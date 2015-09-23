#! /usr/bin/env python

from __future__ import division
import os, sys, numpy as np
import cPickle as pickle
from dpgmm import *
import copy
from scipy.misc import logsumexp
import optparse as op
import lal
import multiprocessing as mp
import copy_reg
import types
import cumulative
import matplotlib
import time
matplotlib.use("MACOSX")

def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)

# ---------------------
# DPGMM posterior class
# ---------------------

class DPGMMGenePosterior(object):
    """
        Dirichlet Process Gaussian Mixture model class
        input parameters:
        
        posterior_samples: posterior samples for which the density estimate needs to be calculated
        
        dimension: the dimensionality of the problem. default = 3
        
        max_stick: maximum number of mixture components. default = 16
        
        bins: number of bins in the d,ra,dec directions. default = [10,10,10]
        
        dist_max: maximum radial distance to consider. default = 218 Mpc
        
        nthreads: number of multiprocessing pool workers to use. default = multiprocessing.cpu_count()
        
        injection: the injection file.
        
        catalog: the galaxy catalog for the ranked list of galaxies
        """
    def __init__(self,posterior_samples,dimension=3,max_sticks=16,nthreads=None):
        self.posterior_samples = np.array(posterior_samples)
        self.dims = dimension
        self.max_sticks = max_sticks
        if nthreads == None:
            self.nthreads = mp.cpu_count()
        else:
            self.nthreads = nthreads
        self.pool = mp.Pool(self.nthreads)

    def _initialise_dpgmm(self):
        self.model = DPGMM(self.dims)
        for point in self.posterior_samples:
            self.model.add([point])
        self.model.setPrior()
        self.model.setThreshold(1e-5)
        self.model.setConcGamma(1.,1.)

    def compute_dpgmm(self):
        self._initialise_dpgmm()
        solve_args = [(nc, self.model) for nc in xrange(1, self.max_sticks+1)]
        solve_results = self.pool.map(solve_dpgmm, solve_args)
        self.scores = np.array([r[1] for r in solve_results])
        self.model = (solve_results[self.scores.argmax()][-1])
        print "best model has ",self.scores.argmax()+1,"components"
        self.density = self.model.intMixture()

    def logPosterior(self,x):
        logPs = [np.log(self.density[0][ind])+prob.logProb(x) for ind,prob in enumerate(self.density[1])]
        return logsumexp(logPs)

    def Posterior(self,x):
        Ps = [self.density[0][ind]*prob.prob(x) for ind,prob in enumerate(self.density[1])]
        return reduce(np.sum,Ps)

# ---------------
# DPGMM functions
# ---------------

def log_cdf(logpdf):
    """
    compute the log cdf from the  log pdf
    
    cdf_i = \sum_i pdf
    log cdf_i = log(\sum_i \exp pdf)
    
    """
    logcdf = np.zeros(len(logpdf))
    logcdf[0] = logpdf[0]
    for j in xrange(1,len(logpdf)):
        logcdf[j]=np.logaddexp(logcdf[j-1],logpdf[j])
    return logcdf-logcdf[-1]

def logPosterior(args):
    density,x = args
    logPs = [np.log(density[0][ind])+prob.logProb(x) for ind,prob in enumerate(density[1])]
    return logsumexp(logPs)

def Posterior(args):
    density,x = args
    Ps = [density[0][ind]*prob.prob(x) for ind,prob in enumerate(density[1])]
    return reduce(np.sum,Ps)

def solve_dpgmm(args):
    (nc, model) = args
    for _ in xrange(nc-1): model.incStickCap()
    try:
        it = model.solve(iterCap=1024)
        return (model.stickCap, model.nllData(), model)
    except:
        return (model.stickCap, -np.inf, model)

# -----------------------
# confidence calculations
# -----------------------

def FindHeights(args):
    (sortarr,cumarr,level) = args
    return sortarr[np.abs(cumarr-np.log(level)).argmin()]

def FindHeightForLevel(inLogArr, adLevels):
    # flatten and create reversed sorted list
    adSorted = np.sort(inLogArr.flatten())[::-1]
    # create a normalized cumulative distribution
    adCum = np.array([logsumexp(adSorted[:i+1]) for i in xrange(len(adSorted))])
    adCum -= adCum[-1]
    # find values closest to levels
    adHeights = []
    adLevels = np.ravel([adLevels])
    for level in adLevels:
        idx = (np.abs(adCum-np.log(level))).argmin()
        adHeights.append(adSorted[idx])
    adHeights = np.array(adHeights)
    return adHeights

def FindLevelForHeight(inLogArr, logvalue):
    # flatten and create reversed sorted list
    adSorted = np.sort(inLogArr.flatten())[::-1]
    # create a normalized cumulative distribution
    adCum = np.array([logsumexp(adSorted[:i+1]) for i in xrange(len(adSorted))])
    adCum -= adCum[-1]
    # find index closest to value
    idx = (np.abs(adSorted-logvalue)).argmin()
    return np.exp(adCum[idx])

#---------
# plotting
#---------

fig_width_pt = 3*246.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
params = {'backend': 'TkAgg',
    'axes.labelsize': 32,
    'text.fontsize': 28,
    'legend.fontsize': 28,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'text.usetex': False,
    'figure.figsize': fig_size}

def parse_to_list(option, opt, value, parser):
    """
    parse a comma separated string into a list
    """
    setattr(parser.values, option.dest, value.split(','))

#-------------------
# start the program
#-------------------

if __name__=='__main__':
#    parser = op.OptionParser()
#    parser.add_option("-i", "--input", type="string", dest="input", help="Input file")
#    parser.add_option("-o", "--output", type="string", dest="output", help="Output file")
#    parser.add_option("--max-stick", type="int", dest="max_stick", help="maximum number of gaussian components")
#    parser.add_option("--threads", type="int", dest="nthreads", help="number of threads to spawn", default=None)
#    parser.add_option("--plots", type="string", dest="plots", help="produce plots", default=False)
#    (options, args) = parser.parse_args()
#
#    input_file = options.input
#    out_dir = options.output
#    os.system('mkdir -p %s'%(out_dir))
#  
#    samples = np.genfromtxt(input_file,names=True)


    samples = []
    for _ in xrange(10000):
        u = np.random.uniform(0.0,1.0)
        if u < 0.3:
            samples.append(np.random.normal(-20.0,2.0))
        elif u < 0.5:
            samples.append(np.random.normal(5.0,0.2))
        else:
            samples.append(np.random.normal(0.0,1.0))
    dpgmm = DPGMMGenePosterior(samples,dimension=1,
                               max_sticks=8,#options.max_stick
                               nthreads=None)#options.nthreads
    dpgmm.compute_dpgmm()

    x = np.linspace(-30.0,30.0,1000)
    px = np.array([dpgmm.logPosterior(xi) for xi in x])

    for ind,prob in enumerate(dpgmm.density[1]):
        print "weight:",dpgmm.density[0][ind],"dof:",prob.getDOF(),"mean:",prob.getLoc(),"scale:",prob.getScale()

    import matplotlib.pyplot as plt
    plt.hist(samples,bins=100,normed=True)
    plt.plot(x,np.exp(px),'k',linewidth=3)
    plt.show()

    y = [0.0]

    print dpgmm.model.stickProb(y)