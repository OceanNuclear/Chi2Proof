from numpy import array as ary
from numpy import log as ln
import numpy as np
from numpy import sqrt, exp
from numpy import sin, cos, tan, arcsin, arccos, arctan
from numpy import pi; tau = pi*2
from matplotlib import pyplot as plt
from referencedata.constants import ENERGY_REGIONS_COARSE, ENERGY_REGIONS
CUSTOM_REGIONS = {'slow':(0.0, 1E5), 'fast':(1E5, 2E7)}
from referencedata.rebin import Rebin
rebinner = Rebin()
from unfoldingsuite.regularization import SmartRegularizer
from unfoldingsuite.externalcodes import UMG33FileController
from unfoldingsuite.maximumentropy import MAXED, IMAXED
from unfoldingsuite.nonlinearleastsquare import GRAVEL, SAND_II, SPUNIT
from unfoldingsuite.pseudoinverse import PseudoInverse
from unfoldingsuite.datahandler import UnfoldingDataHandler
from numpy.random import normal, lognormal, poisson
from numpy.linalg import det, svd, pinv, norm, inv

def read(file_name):
    """Simple function for reading file"""
    with open(file_name) as f:
        data = f.readlines()
    return [float(line.strip()) for line in data if len(line.strip())>0]

def cofactor(A):
    """Taken from stackoverflow"""
    U,sigma,Vt = svd(A)
    N = len(sigma)
    g = np.tile(sigma,N)
    g[::(N+1)] = 1
    G = np.diag(-(-1)**N*np.product(np.reshape(g,(N,N)),1)) 
    return U @ G @ Vt

def cofactor_bottom(A):
    """Takes in square matrix, output the cofactors at the bottom row"""
    ignored_bottom = A[:-1] 
    vector = []
    for elem in range(len(A)):
        matrix = ary([row[:elem].tolist() + row[elem+1:].tolist() for row in ignored_bottom])
        element = (-1)**(len(A)-1+elem) * det(matrix)
        vector.append(element)
    return vector

def cross_prod(*vec):
    assert len(vec)==len(vec[0])-1, "There must be n-1 vectors provided to find a unique orthorgonal direction to all of them in n dimensions"
    dummy_vector = np.ones(len(vec[0]))
    cross_prod_matrix = ary([*vec, dummy_vector])
    assert cross_prod_matrix.shape[0] == cross_prod_matrix.shape[1]
    # cofactor_mat = cofactor(cross_prod_matrix)
    # return cofactor_mat[-1].copy()
    return cofactor_bottom(cross_prod_matrix)

def normalize(vec):
    return vec/sqrt(sum([i**2 for i in vec]))

def turn_into_mult_factors(bin_bounds, lower_E, upper_E, PUL=True):
    """A descriptor of what faction of the old bin structure coincides with the new bin structure."""
    mask = [(tf1 and tf2) for (tf1, tf2) in zip(lower_E<=ary(bin_bounds), ary(bin_bounds)<upper_E)]
    factors = ary([(tf1 and tf2) for tf1, tf2 in zip(mask[:-1], mask[1:])], dtype=float)
    for i in np.where(np.diff(mask))[0]:
        bin_size = ln(bin_bounds[i+1]/bin_bounds[i]) if PUL else (bin_bounds[i+1]-bin_bounds[i])
        dest_lower = max([bin_bounds[i], lower_E])
        dest_upper = min([bin_bounds[i+1], upper_E])
        dest_size = ln(dest_upper/dest_lower) if PUL else (dest_upper-dest_lower)
        factors[i] = dest_size/bin_size
    return factors

def custom_rebin_down(response, flux):
    """Instead of re-binning the response matrix as follows, we """
    assert len(flux)==len(response), "The weight factor(flux) must have the same shape as the response"
    unified_response = response @ ary(flux)/sum(flux) # radionuclide per one unit flux
    return unified_response

def parametric(anchor, parametric_dir, coincident_point):
    output_values = []
    for basis, value in enumerate(coincident_point):
        param = (value - anchor[basis])/parametric_dir[basis]
        output_values.append(anchor + param*parametric_dir)
    return output_values

def choose_point(points, lower_bounds, upper_bounds):
    """Choose, among a list of points all on the same line, the one that may be plotted"""
    deviations = []
    for p in points:
        if ary([lower_bounds<=p, p<=upper_bounds], dtype=bool).all():
            return p
        # calculate the fractional deviation from the upper/lower value
        fractional_dev = (np.clip(p, lower_bounds, upper_bounds) - p)/(upper_bounds - lower_bounds)
        deviations.append(fractional_dev**2)
    # if none of them lies in bound, choose the one that deviate from it the least
    return points[np.argmin(deviations)]

def plot_chi2_lines(ax, anchor, parametric_dir, lower_bounds, upper_bounds, shift=[]):
    points_on_lower_bounds = parametric(anchor, parametric_dir, lower_bounds)
    points_on_upper_bounds = parametric(anchor, parametric_dir, upper_bounds)
    lower_point = choose_point(points_on_lower_bounds, lower_bounds, upper_bounds)
    upper_point = choose_point(points_on_upper_bounds, lower_bounds, upper_bounds)
    plt.plot(*ary([lower_point, upper_point]).T, label='chi^2=0 line', color='black')
    for vector in shift:
        plt.plot(*ary([lower_point+ary(vector), upper_point+ary(vector)]).T)
    return lower_point, upper_point

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
    

REGULARIZATION = True
def unfold_by_alg(unfolder, ap_points, target_chi2, REGULARIZATION=REGULARIZATION):
    """
    unfold by the following algorithms.
    The unfolder is expected to already contain the response_matrix, reaction_rates, response_matrix
    """
    ps_solution, reg_solution = [], []
    umg_MAXED, umg_GRAVEL = [], []
    imaxed_sol = []
    for ap in ap_points:
        unfolder.set_vector('a_priori', ap.tolist())
        ps = PseudoInverse.from_UnfoldingDataHandler(unfolder)
        ps.run_til_chi2_below(target_chi2)
        ps_solution.append(ps.phi[-1].copy())
        if target_chi2<1 and REGULARIZATION:
            try:
                reg = SmartRegularizer.from_UnfoldingDataHandler(unfolder)
                reg.default_run()
                reg_solution.append(reg.phi[-1].copy())
            except:
                pass
        umg = UMG33FileController(base_path+'maxed_te', base_path+'gravel_te')
        umg.verbosity = 0
        umg.set_matrices_and_vectors(unfolder)
        umg.run_gravel(chi_squared_per_degree_of_freedom=target_chi2/len(ap), CLOBBER=True)
        umg_GRAVEL.append(umg.vectors['solution'].copy())
        umg.run_maxed(T=1, TR=0.85, chi_squared_per_degree_of_freedom=target_chi2/len(ap), CLOBBER=True)
        umg_MAXED.append(umg.vectors['solution'].copy())
        imaxed.set_matrices_and_vectors(unfolder)
        imaxed.run()
        imaxed_sol.append(imaxed.vectors['solution'].copy())
    return ps_solution, reg_solution, umg_MAXED, umg_GRAVEL, imaxed_sol

def howfar_4_onechi2(direction, sigma_or_covar, response):
    """How far to walk in this direction in order to get"""
    if np.ndim(sigma_or_covar)==1:
        covar = np.diag(sigma_or_covar**2)
    elif np.ndim(sigma_or_covar)==2:
        covar = sigma_or_covar
    inv_covar = inv(covar)
    v = ary(direction)
    inv_n_squared = ((response@v) @ inv_covar @ (response@v)) # should be a scalar
    n = 1/sqrt(inv_n_squared)
    return n*v

NUMBER_OF_EXPERIMENTS = 10 # choose the number of times this experiment is ran,
AP_SIZE = 100 # and the number of unfolding run per experiment
SHOW2D, SHOW3D = [False, True]
if __name__=='__main__':
    apriori = read('APriori.csv')
    gs = read('gs.csv')
    R1 = read('R1-Si30-102.csv')
    R2 = read('R2-Sc45-102.csv')
    true_flux = read('UKAEA_100_JET-FW.csv')

    small_response_matrix = [[]]
    large_response_matrix = [[], []]
    rnames = ['Si30-102->Si31-beta->P31*-gamma->P31, t1/2=157.36 m ',
            'Sc45-102->Sc46-beta->Ti46*-gamma->Ti46, t1/2=83.79 d']
                # 102 = neutorn capture, releasing a prompt gamma
    for E_range in ENERGY_REGIONS_COARSE.values():
        in_range_flux = turn_into_mult_factors(gs, E_range[0], E_range[1]) * true_flux
        response1_at_this_bin = custom_rebin_down(R1, in_range_flux)
        response2_at_this_bin = custom_rebin_down(R2, in_range_flux)
        large_response_matrix[0].append(response1_at_this_bin.copy())
        large_response_matrix[1].append(response2_at_this_bin.copy())
    tmp_R = ary(large_response_matrix)
    large_response_matrix = ((tmp_R.T/tmp_R.sum(axis=1)).T * tmp_R.sum(axis=1).max()).tolist()
    # large_response_matrix = [[1,0,0], [0,1,0]]

    for E_range in CUSTOM_REGIONS.values():
        in_range_flux = turn_into_mult_factors(gs, E_range[0], E_range[1]) * true_flux
        response1_at_this_bin = custom_rebin_down(R1, in_range_flux)
        small_response_matrix[0].append(response1_at_this_bin)
    base_path = '/home/ocean/Documents/GitHubDir/unfoldinggroup/unfolding/unfoldingsuite/UMG3.3_source/umg_test/'

# 2D demonstration part
    if SHOW2D:
        bin_struct = [E[0] for E in CUSTOM_REGIONS.values()]+[list(CUSTOM_REGIONS.values())[-1][-1]] # get the bin-structure, useful later
        rebinned_ap = rebinner.re_bin(apriori, gs, bin_struct)

        # simulate reaction rates measurements
        results_from_different_exp = []
        for measuring in range(NUMBER_OF_EXPERIMENTS):
            new_flux = normal(true_flux, sqrt(true_flux))
            rebinned_new_flux = rebinner.re_bin(new_flux, gs, bin_struct)
            total_new_pop = small_response_matrix @ ary(rebinned_new_flux)
            results_from_different_exp.append(normal(total_new_pop, sqrt(total_new_pop)))
        #get the true spectrum in this group structure, for reference
        rebinned_true_flux = rebinner.re_bin(true_flux, gs, bin_struct)

        for meas in results_from_different_exp:
            unf = UnfoldingDataHandler(verbosity=0)
            imaxed = IMAXED(verbosity=0)
            unf.set_matrix('response_matrix', small_response_matrix)
            unf.set_vector('reaction_rates', meas.tolist())
            unf.set_vector_uncertainty('reaction_rates', (1/sqrt(meas)).tolist())
            unf.set_vector('group_structure', bin_struct)
            # ap_points = normal(rebinned_ap, sqrt(rebinned_ap)*1000, size=(AP_SIZE, len(rebinned_new_flux)))
            ap_points = normal(rebinned_true_flux, sqrt(rebinned_true_flux)*100000, size=(AP_SIZE, len(rebinned_new_flux)))
            ap_points = np.clip(ap_points, 0, None)

            #using the same set of sample of apriori and measurements, 
            for target_chi2 in [3, 1, 1E-6]:
                #create new lists to contain the solutions
                ps_solution, reg_solution, umg_MAXED, umg_GRAVEL, imaxed_sol = unfold_by_alg(unf, ap_points, target_chi2)
                for solution_list, name in zip([ps_solution, umg_GRAVEL, umg_MAXED, imaxed_sol], ['PseudoInverse', 'umg_GRAVEL', 'umg_MAXED (probably broken)', 'IMAXED']):
                    print([sqrt(sum((((ary(small_response_matrix)@sol_i)- meas)/sqrt(meas))**2)) for sol_i in solution_list])
                    fig, ax = plt.subplots()
                    ax.scatter(*ap_points.T, label='a priori')
                    ax.scatter(*ary(solution_list).T, label='unfolded solution', marker='x')
                    ax.scatter(*rebinned_true_flux, label='True spectrum', marker='+')
                    [getattr(ax, 'set_'+axis+'label')(label) for axis, label in zip(['x','y'], CUSTOM_REGIONS.keys())] #list-comp to add labels onto axes
                    plot_chi2_lines(ax, pinv(small_response_matrix) @ meas, normalize(cross_prod(*small_response_matrix)),
                                    *ary([ax.get_xlim(), ax.get_ylim()]).T)
                    plt.title(name+' chi^2 set to '+str(target_chi2)); plt.legend(); plt.show(); plt.close()
                    input()
                if REGULARIZATION:
                    fig, ax = plt.subplots()
                    ax.scatter(*ap_points.T, label='a priori')
                    ax.scatter(*ary(reg_solution).T, label='unfolded solution', marker='x')
                    ax.scatter(*rebinned_true_flux, label='True spectrum', marker='+')
                    [getattr(ax, 'set_'+axis+'label')(label) for axis, label in zip(['x','y'], CUSTOM_REGIONS.keys())] #list-comp to add labels onto axes
                    plot_chi2_lines(ax, pinv(small_response_matrix) @ meas, normalize(cross_prod(*small_response_matrix)),
                                    *ary([ax.get_xlim(), ax.get_ylim()]).T)
                    plt.title('regularization'); plt.legend(); plt.show(); plt.close()
                    input()
                input()

# 3D demonstration part
    if SHOW3D:
        from mpl_toolkits.mplot3d import Axes3D
        bin_struct = [E[0] for E in ENERGY_REGIONS_COARSE.values()]+[list(ENERGY_REGIONS_COARSE.values())[-1][-1]]
        rebinned_ap  = rebinner.re_bin(apriori, gs, bin_struct)

        #simulate reaction rates measurements
        results_from_different_exp = []
        for measuring in range(NUMBER_OF_EXPERIMENTS):
            new_flux = normal(true_flux, sqrt(true_flux))
            rebinned_new_flux = rebinner.re_bin(new_flux, gs, bin_struct)
            total_new_pop = large_response_matrix @ ary(rebinned_new_flux)
            results_from_different_exp.append(normal(total_new_pop, sqrt(total_new_pop)))
        #get teh true spectrum in this group structure, for reference
        rebinned_true_flux = rebinner.re_bin(true_flux, gs, bin_struct)

        np.random.seed(1)
        for meas in results_from_different_exp:
            unf = UnfoldingDataHandler(verbosity=0)
            imaxed = IMAXED(verbosity=0)
            unf.set_matrix('response_matrix', large_response_matrix)
            unf.set_vector('reaction_rates', meas.tolist())
            unf.set_vector_uncertainty('reaction_rates', (1/sqrt(meas)).tolist())
            unf.set_vector('group_structure', bin_struct)
            ap_points = normal(rebinned_true_flux, sqrt(rebinned_true_flux)*0.01, size=(AP_SIZE, len(rebinned_new_flux)))
            ap_points = np.clip(ap_points, 0, None)

            for target_chi2 in [10, 2, 1, 1E-6]:
                ps_solution, reg_solution, umg_MAXED, umg_GRAVEL, imaxed_sol = unfold_by_alg(unf, ap_points, target_chi2)
                for solution_list, name in zip([ps_solution, umg_GRAVEL, umg_MAXED, imaxed_sol], ['PseudoInverse', 'umg_GRAVEL', 'umg_MAXED (probably broken)', 'IMAXED']):
                    print([sqrt(sum((((ary(large_response_matrix)@sol_i)- meas)/sqrt(meas))**2)) for sol_i in solution_list])
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(*ap_points.T, label='a priori')
                    ax.scatter(*ary(solution_list).T, label='unfolded solution', marker='x')
                    ax.scatter(*rebinned_true_flux, label='True spectrum', marker='+')
                    lower_point, upper_point = plot_chi2_lines(ax, pinv(large_response_matrix) @ meas,
                                    normalize(cross_prod(*large_response_matrix)),
                                    *ary([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]).T,
                                    shift=[howfar_4_onechi2(ary(v), ary([sigma]), ary([[v],])).flatten() for v, sigma in zip(large_response_matrix, sqrt(meas))])
                    [getattr(ax, 'set_'+axis+'label')(label) for axis, label in zip(['x','y','z'], ENERGY_REGIONS_COARSE.keys())] #list-comp to add labels onto axes
                    equichi_circle = rotate_around_axis( np.cross(large_response_matrix[0], large_response_matrix[1]), sqrt(meas), large_response_matrix, 5000)
                    for chi2_value in range(1, 4):
                        ax.plot(*(equichi_circle*chi2_value+(lower_point+upper_point)/2).T,
                            label='chi^2='+str(chi2_value)+' line')
                    plt.title(name+' chi^2 set to '+str(target_chi2)); plt.legend(); plt.show(); plt.close()
                    input()
                    # should draw the lines where chi2 = 1 for that particular reaction instead
                if REGULARIZATION:
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(*ap_points.T, label='a priori')
                    ax.scatter(*ary(reg_solution).T, label='unfolded solution', marker='x')
                    ax.scatter(*rebinned_true_flux, label='True spectrum', marker='+')
                    [getattr(ax, 'set_'+axis+'label')(label) for axis, label in zip(['x','y','z'], ENERGY_REGIONS_COARSE.keys())] #list-comp to add labels onto axes
                    lower_point, upper_point = plot_chi2_lines(ax, pinv(small_response_matrix) @ meas,
                                    normalize(cross_prod(*small_response_matrix)),
                                    *ary([ax.get_xlim(), ax.get_ylim()]).T)
                    plt.title('regularization'); plt.legend(); plt.show(); plt.close()
                    input()
                input()