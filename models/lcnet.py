import sys
sys.path.append('/home/rkohli/aml_projects/src/config_generators/')

from .DNGO import DG

import ConfigSpace
import numpy as np
import threading


from hpbandster.core.base_config_generator import base_config_generator

class LCNetWrapper(base_config_generator):
    def __init__(self,
                 configspace,
                 max_budget,
                 n_points=2000,
                 delta=1.0,
                 n_candidates=1024,
                 min_points_in_model = None,
                 **kwargs):
        """
        Parameters:
        -----------
        directory: string
            where the results are logged
        logger: hpbandster.utils.result_logger_v??
            the logger to store the data, defaults to v1
        overwrite: bool
            whether or not existing data will be overwritten
        """

        super(LCNetWrapper, self).__init__(**kwargs)

        self.n_candidates = n_candidates
        self.model = DG(num_epochs=10000, learning_rate=1e-2, H=50, D=10)

        self.min_points_in_model = min_points_in_model
        self.config_space = configspace

        if min_points_in_model is None:
        	self.min_points_in_model = len(self.config_space.get_hyperparameters())+1
        if self.min_points_in_model < len(self.config_space.get_hyperparameters())+1:
            self.logger.warning('Invalid min_points_in_model value. Setting it to %i'%(len(self.config_space.get_hyperparameters())+1))
            self.min_points_in_model =len(self.config_space.get_hyperparameters())+1
		
        hps = self.config_space.get_hyperparameters()

        self.vartypes = []
        for h in hps:
            if hasattr(h, 'sequence'):
                raise RuntimeError('This version on BOHB does not support ordinal hyperparameters. Please encode %s as an integer parameter!'%(h.name))
            
            if hasattr(h, 'choices'):
                self.vartypes +=[ len(h.choices)]
            else:
                self.vartypes +=[0]
                
        self.vartypes = np.array(self.vartypes, dtype=int)

        self.max_budget = max_budget
        self.train = None
        self.train_targets = None
        self.n_points = n_points
        self.is_trained = False
        self.counter = 0
        self.delta = delta
        
        self.configs = dict()
        self.losses = dict()
        self.good_config_rankings = dict()
        

    def get_config(self, budget):
        """
            function to sample a new configuration
            This function is called inside Hyperband to query a new configuration
            Parameters:
            -----------
            budget: float
                the budget for which this configuration is scheduled
            returns: config
                should return a valid configuration
        """
        
        self.logger.debug('start sampling a new configuration.')

        if not self.is_trained:
            c = self.config_space.sample_configuration().get_array()
        else:
            candidates = np.array([self.config_space.sample_configuration().get_array()
                                   for _ in range(self.n_candidates)])

            # We are only interested on the asymptotic value
            projected_candidates = np.concatenate((candidates, np.ones([self.n_candidates, 1])), axis=1)

            # Compute the upper confidence bound of the function at the asymptote
            m, v = self.model.predict(projected_candidates)

            ucb_values = m + self.delta * np.sqrt(v)
            print(ucb_values)
            # Sample a configuration based on the ucb values
            p = np.ones(self.n_candidates) * (ucb_values / np.sum(ucb_values))
            idx = np.random.choice(self.n_candidates, 1, False, p)

            c = candidates[idx][0]

        sample = ConfigSpace.Configuration(self.config_space, vector=c)

        return sample, _

    def impute_conditional_data(self, array):
        
        return_array = np.empty_like(array)
        
        for i in range(array.shape[0]):
            datum = np.copy(array[i])
            nan_indices = np.argwhere(np.isnan(datum)).flatten()
            
            while (np.any(nan_indices)):
                nan_idx = nan_indices[0]
                valid_indices = np.argwhere(np.isfinite(array[:,nan_idx])).flatten()
                
                if len(valid_indices) > 0:
                    # pick one of them at random and overwrite all NaN values
                    row_idx = np.random.choice(valid_indices)
                    datum[nan_indices] = array[row_idx, nan_indices]
                
                else:
                    # no good point in the data has this value activated, so fill it with a valid but random value
                    t = self.vartypes[nan_idx]
                    if t == 0:
                        datum[nan_idx] = np.random.rand()
                    else:
                        datum[nan_idx] = np.random.randint(t)
                        
                nan_indices = np.argwhere(np.isnan(datum)).flatten()
            return_array[i,:] = datum
        return(return_array)

    def new_result(self, job, update_model=True):
        """
            function to register finished runs
            Every time a run has finished, this function should be called
            to register it with the result logger. If overwritten, make
            sure to call this method from the base class to ensure proper
            logging.
            Parameters:
            -----------
            job_id: dict
                a dictionary containing all the info about the run
            job_result: dict
                contains all the results of the job, i.e. it's a dict with
                the keys 'loss' and 'info'
        """
        super().new_result(job)

        if job.result is None:
			# One could skip crashed results, but we decided to
			# assign a +inf loss and count them as bad configurations
	        loss = np.inf
        else:
			# same for non numeric losses.
			# Note that this means losses of minus infinity will count as bad!
	        loss = job.result["loss"] if np.isfinite(job.result["loss"]) else np.inf

        budget = job.kwargs["budget"]

        if budget not in self.configs.keys():
	        self.configs[budget] = []
	        self.losses[budget] = []

        conf = ConfigSpace.Configuration(self.config_space, job.kwargs["config"])
        
        self.configs[budget].append(conf.get_array())
        self.losses[budget].append(loss)

		
		# skip model building:
		#		a) if not enough points are available
        if len(self.configs[budget]) <= self.min_points_in_model-1:	
            self.logger.debug("Only %i run(s) for budget %f available, need more than %s -> can't build model!"%(len(self.configs[budget]), budget, self.min_points_in_model+1))
            return

		#		b) during warnm starting when we feed previous results in and only update once
        if not update_model:
            return
        
        train_configs = self.impute_conditional_data(np.array(self.configs[budget]))
        train_losses = self.impute_conditional_data(np.array(self.losses[budget]))

        if self.train is None:
            self.train = train_configs
            self.train_targets = train_losses
        else:
            self.train = np.append(self.train, train_configs, axis=0)
            self.train_targets = np.append(self.train_targets, train_losses, axis=0)

        self.model.train(self.train, self.train_targets)
        self.is_trained = True
        self.counter = 0