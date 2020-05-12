from numpy import array as ary; from numpy import log as ln
from numpy import cos, sin, pi, sqrt, exp, arccos;
from scipy.stats import chi2
tau = 2*pi
import numpy as np;
from matplotlib import pyplot as plt
import seaborn as sns

def set_fig():
    sns.set()    
    fig, ax = plt.subplots()
    ax.set_ylabel('Probability distribution function (PDF)')
    fig.set_tight_layout(True)
    return fig, ax

def save(name):
    plt.savefig(name, dpi=200)

if __name__=='__main__':
    fig, ax = set_fig()
    xrange = np.linspace(0, 8, 300)
    xshort = np.linspace(0.4, 8, 300)
    for k in [1,2,3,4,6,9]:
        if k==1:
            ax.plot(xshort, chi2(k).pdf(xshort), label=k)
        else:
            ax.plot(xrange, chi2(k).pdf(xrange), label=k)
    ax.set_xlabel(r'$\chi^2_k$')
    ax.set_title(r'$\chi^2_k$'+' distributions at k degrees of freedom for some positive integer k')
    ax.legend(title='k=')
    save('Wiki_chi2.png')

    xrange = np.linspace(0, 5, 300)
    xshort = np.linspace(0.1, 5, 300)
    fig, ax = set_fig()
    for k in range(1,10):
        if k==1:
            ax.plot(xshort, chi2(k).pdf(xshort), label=k)
        else:
            ax.plot(xrange, chi2(k).pdf(xrange*k)*k, label=k)
    ax.set_xlabel(r'$\frac{\chi^2_k}{k}$')
    ax.set_title(r'$\frac{\chi^2_k}{k}$'+' distributions at k degrees of freedom for some positive integer k')
    ax.legend(title='k=',)
    save('Normalized_chi2_DoF.png')