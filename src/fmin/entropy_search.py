import logging
import george
import numpy as np
import sys
sys.path.append('/home/rkohli/aml_project/src/solver')
from solver.bo_hb import BayesianOptimization

from robo.priors.default_priors import DefaultPrior
from robo.models.gaussian_process import GaussianProcess
from robo.models.gaussian_process_mcmc import GaussianProcessMCMC
from robo.maximizers.direct import Direct
from robo.maximizers.cmaes import CMAES
from robo.maximizers.differential_evolution import DifferentialEvolution
from robo.acquisition_functions.information_gain import InformationGain
from robo.acquisition_functions.ei import EI
from robo.acquisition_functions.marginalization import MarginalizationGPMCMC
from robo.initial_design import init_latin_hypercube_sampling
from utils import get_config_dictionary

logger = logging.getLogger(__name__)


def entropy_search(objective_function, lower, upper, cs, num_iterations=30,
                   maximizer="direct", model="gp_mcmc",
                   min_budget=0.1, max_budget=1,
                   n_init=3, output_path=None, rng=None):
    """
    Entropy search for global black box optimization problems. This is a reimplemenation of the entropy search
    algorithm by Henning and Schuler[1].

    [1] Entropy search for information-efficient global optimization.
        P. Hennig and C. Schuler.
        JMLR, (1), 2012.

    Parameters
    ----------
    objective_function: function
        The objective function that is minimized. This function gets a numpy array (D,) as input and returns
        the function value (scalar)
    lower: np.ndarray (D,)
        The lower bound of the search space
    upper: np.ndarray (D,)
        The upper bound of the search space
    cs: ConfigSpace
        The configspace for the hyperparameters to be optimised. 
    num_iterations: int
        The number of iterations (initial design + BO)
    maximizer: {"direct", "cmaes", "differential_evolution"}
        Defines how the acquisition function is maximized. NOTE: "cmaes" only works in D > 1 dimensions
    model: {"gp", "gp_mcmc"}
        The model for the objective function.
    n_init: int
        Number of points for the initial design. Make sure that it is <= num_iterations.
    output_path: string
        Specifies the path where the intermediate output after each iteration will be saved.
        If None no output will be saved to disk.
    rng: numpy.random.RandomState
        Random number generator
    Returns
    -------
        dict with all results
    """
    assert upper.shape[0] == lower.shape[0], "Dimension miss match"
    assert np.all(lower < upper), "Lower bound >= upper bound"
    assert n_init <= num_iterations, "Number of initial design point has to be <= than the number of iterations"

    if rng is None:
        rng = np.random.RandomState(np.random.randint(0, 10000))

    cov_amp = 2
    n_dims = lower.shape[0]

    initial_ls = np.ones([n_dims])
    exp_kernel = george.kernels.Matern52Kernel(initial_ls,
                                               ndim=n_dims)
    kernel = cov_amp * exp_kernel

    prior = DefaultPrior(len(kernel) + 1)

    n_hypers = 3 * len(kernel)
    if n_hypers % 2 == 1:
        n_hypers += 1

    if model == "gp":
        gp = GaussianProcess(kernel, prior=prior, rng=rng,
                             normalize_output=False, normalize_input=True,
                             lower=lower, upper=upper)
    elif model == "gp_mcmc":
        gp = GaussianProcessMCMC(kernel, prior=prior,
                                 n_hypers=n_hypers,
                                 chain_length=200,
                                 burnin_steps=100,
                                 normalize_input=True,
                                 normalize_output=False,
                                 rng=rng, lower=lower, upper=upper)
    else:
        print("ERROR: %s is not a valid model!" % model)
        return

    a = InformationGain(gp, lower=lower, upper=upper, sampling_acquisition=EI)

    if model == "gp":
        acquisition_func = a
    elif model == "gp_mcmc":
        acquisition_func = MarginalizationGPMCMC(a)

    if maximizer == "cmaes":
        max_func = CMAES(acquisition_func, lower, upper, verbose=False, rng=rng)
    elif maximizer == "direct":
        max_func = Direct(acquisition_func, lower, upper)
    elif maximizer == "differential_evolution":
        max_func = DifferentialEvolution(acquisition_func, lower, upper, rng=rng)
    else:
        print("ERROR: %s is not a valid function to maximize the acquisition function!" % maximizer)
        return

    bo = BayesianOptimization(objective_function, lower, upper, acquisition_func, gp, max_func,
                              initial_design=init_latin_hypercube_sampling,
                              initial_points=n_init, rng=rng, output_path=output_path, min_budget=min_budget, max_budget=max_budget, cs=cs)

    x_best, f_min = bo.run(num_iterations)

    results = dict()
    results["x_opt"] = get_config_dictionary(x_best, cs)
    results["f_opt"] = f_min
    results["incumbents"] = [get_config_dictionary(inc, cs) for inc in bo.incumbents]
    results["incumbent_values"] = [val for val in bo.incumbents_values]
    results["runtime"] = bo.runtime
    results["overhead"] = bo.time_overhead
    results["X"] = [get_config_dictionary(x.tolist(), cs) for x in bo.X]
    results["y"] = [y for y in bo.y]
    return results
