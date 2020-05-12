from numpy import array as ary; from numpy import log as ln
from numpy import cos, sin, pi, sqrt, exp, arccos;
tau = 2*pi
import numpy as np;
from matplotlib import pyplot as plt
import scipy.stats
from matplotlib.patches import Ellipse
from scipy.special import erf
import matplotlib.transforms as transforms
import sys
# from scipy import linregress

true_m, true_c = 0.4, 0.6
f_true = lambda x: true_m*x +true_c # over the range [0,2]
sigma_true = lambda x: 0.1

def fit(x, y, yerr, cov=False, verbose=False):
    (m, c), residual, _rank, _sv, _rcond = np.polyfit(x, y, w=1/ary(yerr), deg=1, full=True)
    if verbose:
        print("rank=", _rank)
        print("singular values =", _sv)
        print("condition number=", _rcond)
    if cov:
        m_c, cov_matr = np.polyfit(x,y, w=1/ary(yerr), deg=1, cov=cov)
        return m_c, cov_matr
    fit_func = lambda x: m*ary(x) + c
    return fit_func, get_rms_residuals(fit_func, x, y, yerr)

def confidence_ellipse(cov, xmean, ymean, ax, fc, ec, n_std=3.0, **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    cov: 2D numpy array of sidelength=2

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    assert (cov.T==cov).flatten().all(), 'Expected symmetric matrix'
    pearson = cov[0, 1]/sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = sqrt(1 + pearson)
    ell_radius_y = sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=fc,
        edgecolor=ec,
        **kwargs)
    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = sqrt(cov[0, 0]) * n_std

    # calculating the stdandard deviation of y ...
    scale_y = sqrt(cov[1, 1]) * n_std

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(xmean, ymean)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def get_rms_residuals(fit_func, x, y, yerr):
    std_score = (fit_func(ary(x))-y)/ary(yerr)
    chi2 = sum(std_score**2)
    return chi2

def _str(arg):
    if isinstance(arg, float):
        return str(round(arg, 4))
    else:
        return str(arg)

def perc(scalar):
    '''percent_area_under_normal_curve'''
    return erf(scalar/sqrt(2))*100

def expand_range_to_includ_0(x0, x1):
    if all([x0>0, x1>0]):
        x0=0
    elif all([x0<0, x1<0]):
        x1=0
    return x0, x1

if __name__=='__main__':
    SEED = 1
    np.random.seed(SEED)

    sigma_meas = lambda x: 0.1
    x_vals = [0.2, 0.8, 1.2, 1.6, 2.0]

    #plot a single example
    fig, ax = plt.subplots()
    y_record, sigma_meas_record = [], []
    for x in x_vals:
        y = f_true(x)+sigma_true(x)*np.random.normal()
        y_record.append(y)
        sigma_meas_record.append(sigma_meas(x))
    ax.errorbar(x_vals, y_record, sigma_meas_record,fmt='.', elinewidth=1, capsize=5)
    fit_func, chi2_fit_from_meas = fit(x_vals, y_record, sigma_meas_record, verbose=False)
    xlim = ax.get_xlim()
    ax.plot( xlim, fit_func(xlim), label=r"$\chi^2_{fit\leftarrow meas}=$"+_str(chi2_fit_from_meas))
    chi2_true_from_meas = get_rms_residuals(f_true, x_vals, y_record, sigma_true(ary(x_vals)))
    ax.plot( xlim, f_true(ary(xlim)), label=r"$\chi^2_{true\leftarrow meas}=$"+_str(chi2_true_from_meas))
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(r'$f_{true}=0.4x+0.6$')
    plt.savefig('A_linregress.png', dpi=150)
    plt.close()

    #plot of the phase space variation in the example above.
    fig, ax = plt.subplots()
    (m, c), cov = fit(x_vals, y_record, sigma_meas_record, cov=True)
    confidence_ellipse(cov, m, c, ax, n_std=3.0, label=_str(perc(3.0))+'%', fc='C2', ec='C2', alpha=1)
    confidence_ellipse(cov, m, c, ax, n_std=2.0, label=_str(perc(2.0))+'%', fc='C1', ec='C1', alpha=1)
    confidence_ellipse(cov, m, c, ax, n_std=1.0, label=_str(perc(1.0))+'%', fc='C0', ec='C0', alpha=1)
    m_width, c_width = 3.5*sqrt(cov[0,0]), 3.5*sqrt(cov[1,1])
    
    ax.set_xlabel('m')
    ax.set_ylabel('c')
    ax.scatter(m, c, label='chi-squared\n=root-sum-squared of residuals\n=minimum', marker='+', color='black', zorder=4)
    arr = ax.arrow(0,0, m,c,
            capstyle='round', length_includes_head=True, head_width=0.02, color='grey',
            label='path of descent taken\nto approach the final (m,c)')
    # ax.annotate("", (m,c), (0,0),
    #         arrowprops=dict(
    #         arrowstyle='->', #length_includes_head=True, head_width=0.02,
    #         label='path of descent taken\nto approach the final (m,c)'))
    leg = ax.legend(title='Confidence bounds')
    handles, labels = ax.get_legend_handles_labels()
    handles[:3], labels[:3] = handles[2::-1], labels[2::-1]
    handles.append(arr)
    labels.append('path of descent taken\nto achieve the final (m,c)')
    ax.legend(title='Confidence bounds', handles=handles, labels=labels,
            loc='lower center')                                                 #  This loc argument may need to be changed if a different chi2 landsacpe is used.
    ax.set_title('negative-log-likelihood surface')
    # ax.set_title('Root sum square')
    ax.set_xlim(*expand_range_to_includ_0(m-m_width, m+m_width))
    ax.set_ylim(*expand_range_to_includ_0(c-c_width, c+c_width))
    plt.savefig('Phase_space.png', dpi=150)
    plt.close()
    #plot 16 examples
    fig, axes = plt.subplots(4,4, sharex=True, sharey=True)
    for ax in axes.flatten():
        y_record, sigma_meas_record = [], []
        for x in x_vals:
            y = f_true(x)+sigma_true(x)*np.random.normal()
            y_record.append(y)
            sigma_meas_record.append(sigma_meas(x))
        ax.errorbar(x_vals, y_record, sigma_meas_record,fmt='.', elinewidth=1, capsize=5)
        fit_func, chi2_fit_from_meas = fit(x_vals, y_record, sigma_meas_record)
        ax.plot( xlim, fit_func(xlim), label=r"$\chi^2_{fit\leftarrow meas}=$"+_str(chi2_fit_from_meas))
        chi2_true_from_meas = get_rms_residuals(f_true, x_vals, y_record, sigma_true(ary(x_vals)))
        ax.plot( xlim, f_true(ary(xlim)), label=r"$\chi^2_{true\leftarrow meas}=$"+_str(chi2_true_from_meas))
        ax.legend()
    fig.set_size_inches(14,10)
    fig.set_tight_layout(True)
    plt.savefig('Ensemble_of_linregress.png', dpi=200)
    plt.close()

    #plot the distribution of chi2 achievable
    chi2_fit_from_meas_record, chi2_true_from_meas_record = [], []
    NUM_SAMPLES = 1000

    fig, ax = plt.subplots()
    for it in range(NUM_SAMPLES):
        y_record, sigma_meas_record = [], []
        for x in x_vals:
            y = f_true(x)+sigma_true(x)*np.random.normal()
            y_record.append(y)
            sigma_meas_record.append(sigma_meas(x))        
        _fit_func, chi2_fit_from_meas = fit(x_vals, y_record, sigma_meas_record)
        chi2_fit_from_meas_record.append(chi2_fit_from_meas)
        std_scores = ( f_true(ary(x_vals))-y_record )/ sigma_true(ary(x_vals))
        chi2_true_from_meas = get_rms_residuals(f_true, x_vals, y_record, sigma_true(ary(x_vals)))
        chi2_true_from_meas_record.append(chi2_true_from_meas)
    ax.hist(chi2_fit_from_meas_record, bins=40, alpha=0.7, label=r"$\chi^2_{fit\leftarrow meas}$")
    ax.hist(chi2_true_from_meas_record, bins=40, alpha=0.7, label=r"$\chi^2_{true\leftarrow meas}$")
    xlim_upper = ax.get_xlim()[1]
    ax.plot(xrange:=np.linspace(0, xlim_upper, 100), scipy.stats.chi2.pdf(x=xrange, df=len(x_vals)-2)*NUM_SAMPLES/2, label=r'Expected $\chi^2$ distribution when ${DoF}=$'+_str(len(x_vals)-2) )
    ax.plot(xrange:=np.linspace(0, xlim_upper, 100), scipy.stats.chi2.pdf(x=xrange, df=len(x_vals))*NUM_SAMPLES/2, label=r'Expected $\chi^2$ distribution when ${DoF}=$'+_str(len(x_vals) ) )
    ax.legend(fontsize='large')
    ax.set_title('Distribution of root-sum-squares of the residuals,\nand the respective distribution they actually follow\n(sampled {} times)'.format(NUM_SAMPLES))
    ax.set_xlabel(r'$\chi^2$, aka root-sum-squares of the residuals')
    plt.savefig('Distribution_of_chi2.png')
    plt.close()