import logging

import numpy as np

from models.DNGO import DG
from robo.acquisition_functions.ei import EI
from robo.acquisition_functions.lcb import LCB
from robo.acquisition_functions.log_ei import LogEI
from robo.acquisition_functions.pi import PI
from robo.maximizers.cmaes import CMAES
from robo.maximizers.differential_evolution import DifferentialEvolution
from robo.maximizers.direct import Direct
from robo.maximizers.random_sampling import RandomSampling
from robo.maximizers.scipy_optimizer import SciPyOptimizer
from utils import get_config_dictionary
from solver.bo_hb import BayesianOptimization
logger = logging.getLogger(__name__)


def bohamiann(objective_function, lower, upper, cs, num_iterations=30, maximizer="direct",
              acquisition_func="log_ei", n_init=3, output_path=None, rng=None):
    """
    Bohamiann uses Bayesian neural networks to model the objective function [1] inside Bayesian optimization.
    Bayesian neural networks usually scale better with the number of function evaluations and the number of dimensions
    than Gaussian processes.

    [1] Bayesian optimization with robust Bayesian neural networks
        J. T. Springenberg and A. Klein and S. Falkner and F. Hutter
        Advances in Neural Information Processing Systems 29

    Parameters
    ----------
    objective_function: function
        The objective function that is minimized. This function gets a numpy array (D,) as input and returns
        the function value (scalar)
    lower: np.ndarray (D,)
        The lower bound of the search space
    upper: np.ndarray (D,)
        The upper bound of the search space
    num_iterations: int
        The number of iterations (initial design + BO)
    acquisition_func: {"ei", "log_ei", "lcb", "pi"}
        The acquisition function
    maximizer: {"direct", "cmaes", "random", "scipy", "differential_evolution"}
        The optimizer for the acquisition function. NOTE: "cmaes" only works in D > 1 dimensions
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
    assert upper.shape[0] == lower.shape[0]
    assert n_init <= num_iterations, "Number of initial design point has to be <= than the number of iterations"

    if rng is None:
        rng = np.random.RandomState(np.random.randint(0, 10000))

    model = DG(num_epochs=20000,
                          learning_rate=0.01, 
                          momentum=0.9,
                          adapt_epoch=5000, 
                          prior=None,
                          H=50,
                          D=10, 
                          alpha=1.0, 
                          beta=1000)

    if acquisition_func == "ei":
        a = EI(model)
    elif acquisition_func == "log_ei":
        a = LogEI(model)
    elif acquisition_func == "pi":
        a = PI(model)
    elif acquisition_func == "lcb":
        a = LCB(model)

    else:
        print("ERROR: %s is not a valid acquisition function!" % acquisition_func)
        return

    if maximizer == "cmaes":
        max_func = CMAES(a, lower, upper, verbose=True, rng=rng)
    elif maximizer == "direct":
        max_func = Direct(a, lower, upper, verbose=True)
    elif maximizer == "random":
        max_func = RandomSampling(a, lower, upper, rng=rng)
    elif maximizer == "scipy":
        max_func = SciPyOptimizer(a, lower, upper, rng=rng)
    elif maximizer == "differential_evolution":
        max_func = DifferentialEvolution(a, lower, upper, rng=rng)

    bo = BayesianOptimization(objective_function, lower, upper, a, model, max_func, cs,
                              initial_points=n_init, output_path=output_path, rng=rng, min_budget=2, max_budget=5)

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
