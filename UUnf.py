from numpy import array as ary; from numpy import log as ln
from numpy import cos, sin, pi, sqrt, exp, arccos;
tau = 2*pi
import numpy as np;
from matplotlib import pyplot as plt
import seaborn as sns
from numpy.linalg import norm
from DoF import _str, perc
from numpy.linalg import pinv

apriori = ary([1.25, 2.6])
rr = 6
true_spec = ary([1,2])
sigma_rr = sqrt(6)
Rm = ary([[1,2]])
sing = 1/norm(Rm[0])*ary([-2,1])
non_sing = 1/norm(Rm[0])*Rm[0]
ap_rr = Rm@apriori
# offset=ary([-1,0.5])
offset= ary([2,1])
# off2 = ary([3.2,0.8])
off2 = ary([3, 0.2])

step = non_sing/norm(Rm)
x_step, y_step = norm(step)**2/step

x_intercept, y_intercept = 6, 3

def set_fig(aspect=True):
    sns.set()
    fig, ax=plt.subplots()
    if aspect: ax.set_aspect(1)
    ax.set_xlabel('Number of neutrons in bin 1'+r'($=\phi_1$)')
    ax.set_ylabel('Number of neutrons in bin 2'+r'($=\phi_2$)')
    return fig, ax

def plot_n_stdev(n, x_intercept, y_intercept, ax, linestyle, **kwargs):
    #upperline
    ax.plot([x_intercept+n*sigma_rr*x_step, 0], [0, y_intercept+n*sigma_rr*y_step], linestyle=linestyle, label=_str(perc(n))+'% confidence bounds', **kwargs)
    ax.plot([x_intercept-n*sigma_rr*x_step, 0], [0, y_intercept-n*sigma_rr*y_step], linestyle=linestyle, **kwargs)
    return ax

def save(name, **kwargs):
    plt.savefig(name, dpi=200, bbox_inches='tight', **kwargs)
    plt.close()

def maxed_func(phi1, neg=True):
    ap_bin1, ap_bin2 = apriori/sum(apriori)
    c = y_intercept + 1*sigma_rr*y_step # upperline
    denom = (0.5*phi1+c)
    phi2 = (c-0.5*phi1)
    entr = 1/denom*(
        phi1 * (ln(phi1/denom) - ln(ap_bin1))
        +phi2* (ln(phi2/denom) - ln(ap_bin2))
    )
    return (entr if not neg else -entr)

arrowarg = dict(arrowprops=dict(arrowstyle='->', color='black'), ha='center', va='center')
arrowarg_short = dict(arrowprops=dict(arrowstyle='->', color='black'))
def skewing_matrix(origin, accelerate=1):
    diag = origin/np.max(origin) # only the largest bin can take the full step
    return accelerate*np.diag(diag)

if __name__=='__main__':
    fig, ax = set_fig()
    ax.plot([x_intercept,0], [0,y_intercept], linestyle='--', color='black', label='measured reaction rates')
    plot_n_stdev(1, x_intercept, y_intercept, ax, linestyle='-', color='C3')
    plot_n_stdev(2, x_intercept, y_intercept, ax, linestyle='-', color='C4')
    # plot_n_stdev(3, 6.0, 3.0, ax, linestyle='-', color='C5')
    ax.scatter(*true_spec, marker='+', label='true spectrum', zorder=100)
    ax.scatter(*apriori, marker='x', label='a priori')
    ax.annotate('singular direction', offset+2*sing, offset, **arrowarg)
    ax.annotate('non-singular direciton', off2+2*non_sing, off2, **arrowarg)
    ax.legend(title='confidence bounds')

    ax.set_title(r'Negative log-likelihood surface in $\mathbf{\phi}$ space')
    # plt.show()
    save('Singular_log_likelihood.png')

    #maxed plot
    phi1 = np.linspace(1, 2, 2000)[1507]
    c = y_intercept + 1*sigma_rr*y_step
    phi2 = c-0.5*phi1
    maxed_sol = ary([phi1, phi2])
    final = apriori + skewing_matrix(apriori, 1.1)@ pinv(Rm)@(rr-Rm@apriori) # shorthand
    fig, ax = set_fig()
    ax.plot([6,0], [0,3], linestyle='--', color='black', label='measured reaction rates')
    plot_n_stdev(1, x_intercept, y_intercept, ax, linestyle='-', color='C3')
    ax.scatter(*true_spec, marker='+', label='true spectrum', zorder=100)
    ax.scatter(*apriori, marker='x', label='a priori')
    #The solution 1
    ax.annotate('', maxed_sol, apriori, **arrowarg)
    ax.scatter(*maxed_sol, label=r'MAXED solution when we set $\frac{\chi^2}{DoF}$=1')
    #solution 0
    ax.annotate('', final, apriori, **arrowarg)
    ax.scatter(*final, label=r'MAXED solution when we set $\frac{\chi^2}{DoF}\rightarrow 0$')
    ax.set_title(r'Path of descent towards lower values of $\chi^2$ taken by MAXED')
    ax.legend()
    save('MAXED_descent.png')

    #GRAVEL plot
    fig, ax = set_fig(aspect=False)
    fig.set_size_inches([10,7.5])
    ax.plot([6,0], [0,3], linestyle='--', color='black', label='measured reaction rates')
    ax.scatter(*true_spec, marker='+', label='true spectrum', zorder=100)
    ax.scatter(*apriori, marker='x', label='a priori', zorder=101)
    plot_n_stdev(1, x_intercept, y_intercept, ax, linestyle='-', color='C3')
    it1 = apriori + skewing_matrix(apriori, 0.45)@ pinv(Rm)@(rr-Rm@apriori)
    it2 = it1 + skewing_matrix(it1, 0.45)@pinv(Rm)@(rr-Rm@it1)
    it3 = it2 + skewing_matrix(it2, 0.45)@pinv(Rm)@(rr-Rm@it2)
    final = apriori + skewing_matrix(apriori, 1.1)@ pinv(Rm)@(rr-Rm@apriori) # shorthand
    ax.annotate('', it1, apriori, ha='left', va='center_baseline', **arrowarg_short)
    ax.annotate('', it2, it1, ha='left', va='center_baseline', **arrowarg_short)
    ax.annotate('', it3, it2, ha='left', va='center_baseline', **arrowarg_short)
    ax.scatter(*apriori, label=r'GRAVEL solution when we set $\frac{\chi^2}{DoF}$=1')
    ax.scatter(*final, label=r'GRAVEL solution when we set  $\frac{\chi^2}{DoF}\rightarrow 0$')
    ax.set_title(r'Path of descent towards lower values of $\chi^2$ taken by GRAVEL')
    ax.legend()
    ax.set_xlim([0,2])
    ax.set_ylim([1,3.5])
    # plt.show()
    save('GRAVEL_descent.png')