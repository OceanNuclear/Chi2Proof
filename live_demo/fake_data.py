# for plotting 3d fake_data
from numpy import array as ary
from numpy import log as ln
import numpy as np
from numpy import sqrt, exp
from numpy import sin, cos, tan, arcsin, arccos, arctan
from numpy import pi; tau = pi*2
from matplotlib import pyplot as plt
from referencedata.constants import ENERGY_REGIONS_COARSE, ENERGY_REGIONS
from unfoldingsuite.regularization import SmartRegularizer
from unfoldingsuite.externalcodes import UMG33FileController
from unfoldingsuite.maximumentropy import MAXED, IMAXED
from unfoldingsuite.nonlinearleastsquare import GRAVEL, SAND_II, SPUNIT
from unfoldingsuite.pseudoinverse import PseudoInverse
from unfoldingsuite.datahandler import UnfoldingDataHandler
from numpy.random import normal # lognormal
from scipy.stats import lognorm, poisson
from numpy.linalg import det, svd, pinv, norm, inv
from mpl_toolkits.mplot3d import Axes3D

R = ary([[4,2,1], [1,3,2]]) # response matrix
phi_true = ary([100,100,100]) # hidden from program
N_true = R @ phi_true # hidden from program
sigma_N_true = sqrt(N_true) # does not account for the poisson distribution nature of the flux itself
# i.e. assume the flux is so high that statistical variation does not matter anymore

base_path = '/home/ocean/Documents/GitHubDir/unfoldinggroup/unfolding/unfoldingsuite/UMG3.3_source/check_behaviour/'
from os.path import curdir, abspath
original_dir = abspath(curdir)

NUM_AP = 40

def unit_vec(vec):
    return vec/norm(vec)

singular_dir = unit_vec(np.cross(R[0], R[1]))

def read_gravel_file():
    with open(base_path+'gravel_o.plo') as f:
        plo = f.readlines()[:]
    for line in plo:
        # print(line)
        pass
    # probably some sort of conversion is needed to convert it into the correct representation.
    return plo

def prepare_graph(ap_list, phi_true):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*ary(ap_list).T, label='a priori')
    ax.scatter(*phi_true, color='black', marker='x', label='true spectrum')
    return fig, ax

def plot_surface(ax, list_of_sol):
    """
    plot the surface that fits list_of_sol best.
    """
    a, b, c = pinv(list_of_sol)@np.ones(len(list_of_sol)) #where ax+by+cz=1 on all pts of the surface.
    xmax, xmin, ymax, ymin, _, _ = ax.get_w_lims()
    def z(x, y):
        """for any point on the plane, plug in x,y, get back z"""
        return (1-a*x-b*y)/c
    #line 1
    x, y = np.meshgrid([xmin, xmax],[ymin, ymax])
    ax.plot_surface(x, y , z(x,y), alpha=0.2)
    return a, b, c

def parametric_within_bounds(anchor, unit_dir, lower_bounds, upper_bounds):
    r"""
    get the coordinates where a parametric line intersects with a cube
    [l] = vector denoting line,
    [a] = anchor vector
    [d] = unit vector pointing // to the line
    l = parametric variable
    [l]:[a]+ \lambda*[d]
    x_min<l[0]<x_max
    ...
    """
    lambda_min, lambda_max = [], []
    for basis in range(3):
        if np.sign(unit_dir[basis])>0:
            lambda_min.append((lower_bounds[basis]-anchor[basis])/unit_dir[basis])
            lambda_max.append((upper_bounds[basis]-anchor[basis])/unit_dir[basis])
        elif np.sign(unit_dir[basis])<0:
            lambda_max.append((lower_bounds[basis]-anchor[basis])/unit_dir[basis])
            lambda_min.append((upper_bounds[basis]-anchor[basis])/unit_dir[basis])

    min_lambda_point = anchor + max(lambda_min) * unit_dir
    max_lambda_point = anchor + min(lambda_max) * unit_dir
    return min_lambda_point, max_lambda_point

def plot_chi2_line(ax, anchor, singular_dir, shift=[], chi2_mark=[1], sigma_N_meas=None, R=None):
    """
    Plot the singular directions within the viewing cube, fixed at an anchor
    """
    lims = ary(ax.get_w_lims())
    lower_bounds, upper_bounds = lims[::2], lims[1::2]
    min_lambda_point, max_lambda_point = parametric_within_bounds(anchor, unit_vec(singular_dir), lower_bounds, upper_bounds)
    ax.plot(*ary([min_lambda_point, max_lambda_point]).T, label='chi^2=0')
    ax.set_xlim3d(lims[0], lims[1]); ax.set_ylim3d(lims[2], lims[3]); ax.set_zlim3d(lims[4], lims[5])
    for chi2_value in chi2_mark:
        equichi_circle = rotate_around_axis(singular_dir, sigma_N_meas, R, num_points=120)
        ax.plot(*(equichi_circle*sqrt(chi2_value) + anchor).T, label='chi^2='+str(chi2_value)+' circle')
    return ax

def finish_graph(title):
    plt.legend()
    plt.title(title)
    plt.show()
    plt.close()

def no_curve_ps(R, S_N_inv, N_meas, phi_current):
    """
    Quick patch for pseudoinverse method that accounts for the errors
    """
    N_current = R @ phi_current
    S_phi_inv = R.T @ S_N_inv @ R
    from numpy.linalg import pinv
    step = pinv(S_phi_inv) @ R.T @ S_N_inv @ (N_meas - N_current)
    return step

def curved_ps(R, S_N_inv, N_meas, phi_current):
    step = no_curve_ps(R, S_N_inv, N_meas, phi_current)
    change_frac = step/phi_current
    scaled_change_frac = exp(change_frac)-1
    scalar_sf = np.clip(1/max(scaled_change_frac), 0,1)
    reduced_change_frac = scaled_change_frac * scalar_sf
    print("Step size used =",scalar_sf)
    new_phi = (reduced_change_frac+1)*phi_current
    return new_phi

def plot_path(ax, *paths):
    """
    Plot the descent path from the a_priori towards the final solution for each algorithm at each unfolding session
    """
    for path in paths:
        ax.plot(*ary(path).T, color='C3')
    return

# indented this block so that I can hide it in sublime text
if True:
    def cartesian_spherical(x, y, z):
        """
        convert a cartesian unit vector into theta-phi representation
        """
        x,y,z = ary(np.clip([x,y,z],-1,1), dtype=float) #change the data type to the desired format
        Theta = arccos(z)
        Phi = arctan(np.divide(y,x))    #This division is going to give nan if (x,y,z) = (0,0,1)
        Phi = np.nan_to_num(Phi)    #Therefore assert phi = 0 if (x,y) = (0,0)
        Phi+= ary( (np.sign(x)-1), dtype=bool)*pi #if x is positive, then phi is on the RHS of the circle; vice versa.
        return ary([Theta, Phi])

    def QuatToR(q):
        theta = 2 * arccos(np.clip(q[0],-1,1))

        R = np.identity(3)
        # if theta>2E-5 or abs(theta-pi)>2E-5: # theta_prime not approaching 0 or pi
        if True:
            R[0][0] -= 2*( q[2]**2  +q[3]**2  )
            R[1][1] -= 2*( q[1]**2  +q[3]**2  )
            R[2][2] -= 2*( q[1]**2  +q[2]**2  )
            R[0][1] -= 2*( q[0]*q[3]-q[1]*q[2])
            R[0][2] -= 2*(-q[0]*q[2]-q[1]*q[3])
            R[1][0] -= 2*(-q[0]*q[3]-q[1]*q[2])
            R[1][2] -= 2*( q[0]*q[1]-q[2]*q[3])
            R[2][0] -= 2*( q[0]*q[2]-q[1]*q[3])
            R[2][1] -= 2*(-q[0]*q[1]-q[2]*q[3])

        return R

    def rotate_around_axis(axis, sigma_or_covar, response, num_points=120, initial_dir = [1,0,0]):
        """
        Specifically generated for the case of reponse matrix with dimension m=2, n=3.
        Plots a circle on the equi-chi^2 surface.
        """
        unit_axis = axis/norm(axis)
        initial_dir -= (unit_axis @ initial_dir) * unit_axis
        list_of_points = []
        for theta in np.linspace(0, tau, num_points):
            R = QuatToR([cos(theta/2), *(unit_axis * sin(theta/2))])
            vector = howfar_4_onechi2(R @ initial_dir, sigma_or_covar, response)
            list_of_points.append(vector)
        return ary(list_of_points)

    def howfar_4_onechi2(direction, sigma_or_covar, response):
        """How far to walk in this direction in order to get one chi2 deviation"""
        if np.ndim(sigma_or_covar)==1:
            covar = np.diag(sigma_or_covar**2)
        elif np.ndim(sigma_or_covar)==2:
            covar = sigma_or_covar
        inv_covar = inv(covar)
        v = ary(direction)
        inv_n_squared = ((response@v) @ inv_covar @ (response@v)) # should be a scalar
        n = 1/sqrt(inv_n_squared)
        return n*v

if __name__=='__main__':

    #create unfolding object to be used
    unf = UnfoldingDataHandler(verbosity=0)
    bin_struct = [E[0] for E in ENERGY_REGIONS_COARSE.values()]+[list(ENERGY_REGIONS_COARSE.values())[-1][-1]]
    unf.set_vector('group_structure', bin_struct)
    unf.set_matrix('response_matrix', R.tolist())
    np.random.seed(0)
    N_meas = ary([normal(population, 0) for population, sigma in zip(N_true, sigma_N_true)])
    # N_meas = ary([normal(population, 0) for population, sigma in zip(N_true, sigma_N_true)])
    sigma_N_meas = sqrt(N_meas)
    unf.set_vector('reaction_rates', N_meas.tolist())
    unf.set_vector_uncertainty('reaction_rates', (sigma_N_meas/N_meas).tolist())
    S_N_inv = inv(np.diag(sigma_N_meas**2))

    def calculate_chi2(phi, S_N_inv=S_N_inv, R=R, N_meas=N_meas):
        return (R@phi - N_meas)@S_N_inv@(R@phi - N_meas)

    def populate_entire_cube(ax, resolution=50):
        bounds = ax.get_w_lims()
        lower_bounds, upper_bounds = bounds[::2], bounds[1::2]
        x,y,z = [coords.flatten() for coords in np.meshgrid(*[np.linspace(lower_bounds[i], upper_bounds[i],resolution) for i in range(3)])]
        return x,y,z

    def calculate_chi2_long(phi_long, sigma_N_meas=sigma_N_meas, R=R, N_meas=N_meas):
        return ((((R@phi_long).T - N_meas) / sigma_N_meas)**2).sum(axis=1)

    def draw_chi2_surface(ax, chi2_value, **kwargs):
        x,y,z = populate_entire_cube(ax)
        mask = np.isclose( calculate_chi2_long(ary([x,y,z])), chi2_value, **kwargs)
        points = ary([x,y,z]).T[mask].T
        ax.scatter(*points, alpha=0.1)

    imaxed_sol, maxed_sol, gravel_sol, ps_sol = [], [], [], [] # containers for the final unfolded solutions
    imaxed_steps, maxed_steps, gravel_steps, ps_steps = [], [], [], [] # containers for the intermediates solutions
    loss_values = []
    target_chi2 = 2
    ap_list=normal(phi_true, sqrt(phi_true), size=[NUM_AP, 3])
    for ap in ap_list:
        if (ap<0).any():
            print("found a negative value. Ignoring...")
            continue
        unf.set_vector('a_priori', ap.tolist())
        imaxed = IMAXED(verbosity=0)
        imaxed.set_matrices_and_vectors(unf)
        imaxed.run(omega=target_chi2)
        imaxed_sol.append(imaxed.vectors['solution'].copy())
        imaxed_steps.append([imaxed.lambda2spectrum(l) for l in imaxed.l_vec])

        umg = UMG33FileController(base_path+'maxed_modpot', base_path+'gravel_te')
        umg.set_matrices_and_vectors(unf)

        umg.run_gravel(plot_intv=1, chi_squared_per_degree_of_freedom=target_chi2/2, delete=False, CLOBBER=True)
        plo = read_gravel_file() # this can, somebow, be translated into gravel_steps, but I don't know how.
        gravel_sol.append(umg.vectors['solution'].copy())

        umg.run_maxed(chi_squared_per_degree_of_freedom=target_chi2/2, CLOBBER=True)
        maxed_sol.append(umg.vectors['solution'].copy())
        maxed_steps.append([imaxed.lambda2spectrum( ary([l1,l2]) + l3/imaxed.unc) for (l1,l2,l3) in umg.l_vec])

        ps = ap+no_curve_ps(R, S_N_inv, N_meas, ap)
        ps_sol.append(ps)

        loss_values.append([
        imaxed.potential_function( pinv(R.T) @ -ln(imaxed.vectors['solution']/ap) ),
        imaxed.potential_function( pinv(R.T) @ -ln(umg.vectors['solution']/ap) ),
        imaxed.potential_function( pinv(R.T) @ -ln(ap/ap) ),
        ]) # for confirming that the loss value of imaxed should be higher than the loss value of umg maxed
    cent = phi_true + no_curve_ps(R, S_N_inv, N_meas, phi_true) # use this point as the center

    '''
    fig, ax = prepare_graph(ap_list, phi_true)
    ax.scatter(*ary(imaxed_sol).T)
    lims = ary(ax.get_w_lims())
    # plot_path(ax, *imaxed_steps)
    plot_chi2_line(ax, cent, singular_dir, chi2_mark=[2,1], sigma_N_meas=sigma_N_meas, R=R)
    ax.set_xlim3d(lims[0], lims[1]); ax.set_ylim3d(lims[2], lims[3]); ax.set_zlim3d(lims[4], lims[5])

    draw_chi2_surface(ax, 2, atol=0.1)
    finish_graph('IMAXED solutions')

    fig, ax = prepare_graph(ap_list, phi_true)
    ax.scatter(*ary(gravel_sol).T)
    plot_chi2_line(ax, cent, singular_dir, chi2_mark=[2,1], sigma_N_meas=sigma_N_meas, R=R)
    plot_path(ax, *gravel_steps)
    draw_chi2_surface(ax, 2, atol=0.1)
    plot_surface(ax, gravel_sol)
    ax.plot(*ary([cent, cent+R[0]]).T, label='R[0]')
    ax.plot(*ary([cent, cent+R[1]]).T, label='R[1]')
    finish_graph('GRAVEL solutions')

    fig, ax = prepare_graph(ap_list, phi_true)
    ax.scatter(*ary(maxed_sol).T)
    plot_chi2_line(ax, cent, singular_dir, chi2_mark=[2,1], sigma_N_meas=sigma_N_meas, R=R)
    plot_path(ax, *maxed_steps)
    # plot_surface(ax, maxed_sol)
    draw_chi2_surface(ax, 2, atol=0.1)
    ax.plot(*ary([cent, cent+R[0]]).T, label='R[0]')
    ax.plot(*ary([cent, cent+R[1]]).T, label='R[1]')
    finish_graph('MAXED solutions')

    fig, ax = prepare_graph(ap_list, phi_true)
    ax.scatter(*ary(ps_sol).T)
    lims = ary(ax.get_w_lims())
    plot_chi2_line(ax, cent, singular_dir, chi2_mark=[2,1], sigma_N_meas=sigma_N_meas, R=R)
    draw_chi2_surface(ax, 2, atol=0.1)
    finish_graph('Pseudo-Inverse (one-step) solutions')
    '''

    #final comparison
    fig, ax = prepare_graph(ap_list, phi_true)
    ax.scatter(*ary(maxed_sol).T, label='maxed solution')
    ax.scatter(*ary(imaxed_sol).T, label='imaxed solution')
    # plot_path(ax, *maxed_steps)
    draw_chi2_surface(ax, 2, atol=0.1)
    plot_chi2_line(ax, cent, singular_dir, chi2_mark=[2,1], sigma_N_meas=sigma_N_meas, R=R)
    finish_graph('IMAXED and MAXED')