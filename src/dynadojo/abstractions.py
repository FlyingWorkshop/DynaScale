import itertools
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

class AbstractModel(ABC):
    def __init__(self, embed_dim: int, timesteps: int, max_control_cost: float, seed: int | None, **kwargs):
        """
        Base class for all models. Your models should subclass this class.

        :param embed_dim: embedded dimension of the dynamics
        :param timesteps: timesteps per training trajectory (per action horizon)
        :param max_control_cost: maximum control cost per trajectory (per action horizon)
        :param kwargs:
        """
        self._embed_dim = embed_dim
        # NOTE: this is the timesteps of the training data; NOT the predicted trajectories
        self._timesteps = timesteps
        self._max_control_cost = max_control_cost
        self._seed = seed

    @abstractmethod
    def fit(self, x: np.ndarray, **kwargs) -> None:
        """
        Trains the model. Your models must implement this method.

        :param x: (n, timesteps, embed_dim) trajectories tensor
        :param kwargs:
        :return: None
        """
        raise NotImplementedError

    def act(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Determines the control for each action horizon. Your models should override this method if they use
        non-trivial control.

        :param x: (n, timesteps, embed_dim) trajectories tensor
        :param kwargs:
        :return: (n, timesteps, embed_dim) controls tensor
        """
        return np.zeros_like(x)

    def act_wrapper(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Wrapper for act() called in evaluate(). Verifies control tensor is the right shape. You should NOT override this.

        :param x: (n, timesteps, embed_dim) trajectories tensor
        :param kwargs:
        :return: (n, timesteps, embed_dim) controls tensor
        """
        control = self.act(x, **kwargs)
        assert control.shape == x.shape
        return control

    @abstractmethod
    def predict(self, x0: np.ndarray, timesteps: int, **kwargs) -> np.ndarray:
        """
        Predict how initial conditions matrix evolves over a given number of timesteps. Your models must implement this method.

        NOTE: The timesteps argument can differ from the ._timesteps attribute. This allows model to train on data
        that is shorter/longer than trajectories in the test set.

        NOTE: The first coordinate of each trajectory should match the initial condition.

        :param x0: (n, embed_dim) initial conditions matrix
        :param timesteps: timesteps per predicted trajectory
        :param kwargs:
        :return: (n, timesteps, embed_dim) trajectories tensor
        """
        raise NotImplementedError

    def predict_wrapper(self, x0: np.ndarray, timesteps, **kwargs) -> np.ndarray:
        """
        Wrapper for predict() called in evaluate(). Verifies predicted trajectories tensor has the right shape.
        You should NOT override this.

        NOTE: Does not enforce that the first coordinate of each trajectory is the same as the initial condition. This
        allows DynaDojo to handle models that completely mispredict trajectory evolution.

        :param x0: (n, embed_dim) initial conditions matrix
        :param timesteps: timesteps per predicted trajectory
        :param kwargs:
        :return: (n, timesteps, embed_dim) trajectories tensor
        """
        pred = self.predict(x0, timesteps, **kwargs)
        n = x0.shape[0]
        assert pred.shape == (n, timesteps, self._embed_dim)
        return pred


class AbstractSystem(ABC):
    def __init__(self, latent_dim, embed_dim, seed: int | None, **kwargs):
        """
        Base class for all systems. Your systems should subclass this class.

        NOTE: The reason we use properties and setter methods for ._latent_dim and ._embed_dim is to allow
        systems to maintain information through parameter shifts. See LDSSystem in './systems/lds.py' for a principled
        usage example of the setter methods.
        """
        self._latent_dim = latent_dim
        self._embed_dim = embed_dim
        self._seed = seed

    @property
    def latent_dim(self):
        return self._latent_dim

    @property
    def embed_dim(self):
        return self._embed_dim

    @latent_dim.setter
    def latent_dim(self, value):
        self._latent_dim = value

    @embed_dim.setter
    def embed_dim(self, value):
        self._embed_dim = value

    @abstractmethod
    def make_init_conds(self, n: int, in_dist=True) -> np.ndarray:
        """
        Abstract method to generate initial conditions. Your system should override this method.

        NOTE: Systems developers determine what counts as in vs out-of-distribution. DynaDojo doesn't provide
        any verification that this distinction makes sense or even exists.

        :param n: number of initial conditions
        :param in_dist: Boolean. If True, generate in-distribution initial conditions. Defaults to True. If False,
        generate out-of-distribution initial conditions.
        :return: (n, embed_dim) initial conditions matrix
        """
        raise NotImplementedError

    def make_init_conds_wrapper(self, n: int, in_dist=True):
        """
        Wrapper for make_init_conds() called in evaluate(). Verifies initial condition matrix is the right shape.
        You should NOT override this.

        :param n: number of initial conditions
        :param in_dist: Boolean. If True, generate in-distribution initial conditions. Defaults to True. If False,
        generate out-of-distribution initial conditions.
        :return: (n, embed_dim) initial conditions matrix
        """
        init_conds = self.make_init_conds(n, in_dist)
        assert init_conds.shape == (n, self.embed_dim)
        return init_conds

    @abstractmethod
    def make_data(self, init_conds: np.ndarray, control: np.ndarray, timesteps: int, noisy=False) -> np.ndarray:
        """
        Abstract method that makes trajectories from initial conditions. Your system must override this method.

        :param init_conds: (n, embed_dim) initial conditions matrix
        :param control: (n, timesteps, embed_dim) controls tensor
        :param timesteps: timesteps per training trajectory (per action horizon)
        :param noisy: Boolean. If True, add noise to trajectories. Defaults to False. If False, no noise is added.
        :return: (n, timesteps, embed_dim) trajectories tensor
        """
        raise NotImplementedError

    def make_data_wrapper(self, init_conds: np.ndarray, control: np.ndarray = None, timesteps: int = 1,
                          noisy=False) -> np.ndarray:
        """
        Wraps make_data(). Checks that trajectories tensor has the proper shape. You should NOT override this.

        :param init_conds: (n, embed_dim) initial conditions matrix
        :param control: (n, timesteps, embed_dim) controls tensor
        :param timesteps: timesteps per training trajectory (per action horizon)
        :param noisy: Boolean. If True, add noise to trajectories. Defaults to False. If False, no noise is added.
        :return: (n, timesteps, embed_dim) trajectories tensor
        """
        assert timesteps > 0
        assert init_conds.ndim == 2 and init_conds.shape[1] == self.embed_dim
        n = init_conds.shape[0]
        if control is None:
            control = np.zeros((n, timesteps, self.embed_dim))
        assert control.shape == (n, timesteps, self.embed_dim)
        data = self.make_data(init_conds=init_conds,
                              control=control, timesteps=timesteps, noisy=noisy)
        assert data.shape == (n, timesteps, self.embed_dim)
        return data

    @abstractmethod
    def calc_error(self, x, y) -> float:
        """
        Calculates the error between two tensors of trajectories. Your systems must implement this.

        :param x: (n, timesteps, embed_dim) trajectories tensor
        :param y: (n, timesteps, embed_dim) trajectories tensor
        :return: Float. The error between x and y.
        """
        raise NotImplementedError

    def calc_error_wrapper(self, x, y) -> float:
        """
        Wraps calc_error. Checks that calc_error is called with properly-shaped x and y.
        Your systems should NOT override this.

        :param x: (n, timesteps, embed_dim) trajectories tensor
        :param y: (n, timesteps, embed_dim) trajectories tensor
        :return: Float. The error between x and y.
        """
        assert x.shape == y.shape
        return self.calc_error(x, y)

    @abstractmethod
    def calc_control_cost(self, control: np.ndarray) -> np.ndarray:
        """
        Calculate the control cost for each control TRAJECTORY (i.e., calculates the costs for every
        control matrix, not for the whole tensor). Your systems must implement this.

        :param control: (n, timesteps, embed_dim) controls tensor
        :return: (n,) control costs vector
        """
        raise NotImplementedError

    def calc_control_cost_wrapper(self, control: np.ndarray) -> np.ndarray:
        """
        Wraps calc_control_cost(). Your systems should NOT override this.

        :param control: (n, timesteps, embed_dim) controls tensor
        :return: (n,) control costs vector
        """
        assert control.shape[2] == self.embed_dim and control.ndim == 3
        cost = self.calc_control_cost(control)
        assert cost.shape == (len(control),)
        return cost


class Challenge:
    def __init__(self,
                 N: list[int],
                 L: list[int],
                 E: list[int] | int | None,
                 t: int,
                 max_control_cost_per_dim: int,
                 control_horizons: int,
                 system_cls: type[AbstractSystem],
                 reps: int,
                 test_examples: int,
                 test_timesteps: int,
                 system_kwargs: dict | None = None,
                #  save_class: bool = False,
                 ):
        """
        :param N: train sizes, (# of trajectories)
        :param L: latent dimensions
        :param E: embedded dimensions. Optional. 
            If list, then evaluate iterates across embedded dimensions. (e >= l)
            If int, then evaluate uses a fixed embedded dimension. (E >= max(L))
            If None, then evaluate sets the embedded dimension equal to the latent dimension. (e = l)
        :param t: timesteps (length of a trajectory)
        :param max_control_cost_per_dim: max control cost per control trajectory
        :param control_horizons: number of times to generate training data with control
        :param system_cls: class constructor (NOT instance) for a concrete system
        :param reps: number of times to repeat each experiment
        :param test_examples: test size
        :param test_timesteps: test timesteps
        :param system_kwargs:
        """
        assert control_horizons >= 0

        self._id = itertools.count()
        self._N = N
        self._L = L
        self._E = E
        self._t = t
        self._max_control_cost_per_dim = max_control_cost_per_dim
        self._system_cls = system_cls
        self._system_kwargs = system_kwargs or {}
        self._control_horizons = control_horizons
        self._reps = reps
        self._test_examples = test_examples
        self._test_timesteps = test_timesteps

    def evaluate(self,
                 model_cls: type[AbstractModel],
                 model_kwargs: dict | None = None,
                 fit_kwargs: dict | None = None,
                 act_kwargs: dict | None = None,
                 ood=False,
                 noisy=False,
                 id=None,
                 num_parallel_cpu=-1,
                 seed=None,
                 # Filters which reps and L to evaluate. If None, no filtering is performed. 
                 # We recommend using these filters to parallelize evaluation across multiple machines, while retaining reproducibility.
                 reps_filter: list[int] = None,
                 L_filter: list[int] | None = None,
                 rep_l_filter: list[tuple[int, int]] | None = None,
                 #  **kwargs
                 ) -> pd.DataFrame:
        """
        Evaluates a model class (NOT an instance) on a dynamical system over a set of experimental parameters.

        :param model_cls: model class to be evaluated
        :param model_kwargs:
        :param fit_kwargs:
        :param act_kwargs:
        :param ood: Boolean. If True, also test on out-distribution initial conditions for the test set. (For FixedError, search is performed on ood_error if ood=True.) Defaults to False.
        If False, generate out-of-distribution initial conditions for the test set.
        :param noisy: Boolean. If True, add noise to train set. Defaults to False. If False, no noise is added.
        :param id: model ID associated with evaluation results in returned DataFrame
        :param num_parallel_cpu: number of cpus to use in parallel. Defaults to -1, which uses all available cpu.
        :param seed: to seed random number generator for seeding systems and models. Defaults to None. Is overriden by seeds in system_kwargs or model_kwargs.
        :param eval_reps: if provided, will only evaluate the given rep_ids. Defaults to None, which evaluates all repetitions.
        :param eval_L: if provided, will only evaluate the given latent dimensions. Defaults to None, which evaluates all latent dimensions.
        :param eval_rep_l: if provided, will only evaluate the given (rep_id, latent_dim) pairs. Defaults to None, which evaluates all (rep_id, latent_dim) pairs.
        :
        return: a pandas DataFrame with experimental results
        """

        model_kwargs = model_kwargs or {}
        fit_kwargs = fit_kwargs or {}
        act_kwargs = act_kwargs or {}


        # Sets embedded dim array when self._E is None, constant, or an array    
        if self._E is None: 
            E = self._L
        elif isinstance(self._E, int):
            assert self._E >= max(self._L), "E must be greater than or equal to max(L)."
            E = [self._E] * len(self._L)
        else:
            assert isinstance(self._E, list), "E must of type List[int], int, or None."
            assert len(self._E) != len(self._L), "E (type List[int]) and L must be of the same length."
            E = self._E

        # Handling which reps to evaluate. 
        ## First, making seeds for all reps
        if seed:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        ## We have to have the model seeded so that it is the same for each iteration of training in system_run (which may vary/search through N)
        system_seeds = rng.integers(0, 2**32, size=self._reps*len(self._L))
        model_seeds = rng.integers(0, 2**32, size=self._reps*len(self._L))

        ## Second creating all system_run arguments (rep_id, latent_dim, embed_dim, system_seed, model_seed)
        system_run_args = zip(itertools.product(range(self._reps), zip(self._L, E)), system_seeds, model_seeds)
        ## flatten system_args to a list of tuples
        system_run_args = [(r, l, e, system_seed, model_seed) for (r, (l, e)), system_seed, model_seed in system_run_args]
        ## Third, figuring out which reps to run based on specified subset of reps
        if reps_filter is not None and len(reps_filter)> 0:
            system_run_args = [args for args in system_run_args if args[0] in reps_filter]
        ## Fourth, figuring out which systems to run based on specified subset of L
        if L_filter is not None and len(L_filter)> 0:
            system_run_args = [args for args in system_run_args if args[1] in L_filter]
        ## Fifth, figuring out which systems to run based on specified subset of (rep_id, L)
        if rep_l_filter is not None and len(rep_l_filter)> 0:
            system_run_args = [args for args in system_run_args if (args[0], args[1]) in rep_l_filter]

        fixed_run_args = { 
            # **kwargs, #ToDo: consider adding extra kwargs to pass to system_run
            "model_cls" : model_cls, 
            "model_kwargs" : model_kwargs, 
            "fit_kwargs": fit_kwargs, 
            "act_kwargs":act_kwargs, 
            "noisy":noisy, 
            "test_ood": ood 
        }

        # Run systems in parallel
        data = Parallel(n_jobs=num_parallel_cpu, timeout=1e6)(
            delayed(self.system_run)(rep_id, l, e, **fixed_run_args ,system_seed=system_seed, model_seed=model_seed) 
            for rep_id, l, e, system_seed, model_seed in system_run_args)

        if data:
            data = pd.concat(data)
            data["id"] = id or next(self._id)
            data["control_horizon"] = self._control_horizons
        return data
        
    def _gen_trainset(self, system, n: int, timesteps: int, noisy=False):
        train_init_conds = system.make_init_conds_wrapper(n)
        return system.make_data_wrapper(train_init_conds, timesteps=timesteps, noisy=noisy)

    def _gen_testset(self, system, in_dist=True):
        test_init_conds = system.make_init_conds_wrapper(self._test_examples, in_dist)
        return system.make_data_wrapper(test_init_conds, timesteps=self._test_timesteps)

    def _fit_model(self, system, model, x: np.ndarray, timesteps: int,  max_control_cost: int,  fit_kwargs: dict = None,
                   act_kwargs: dict = None, noisy=False) -> int:
        total_cost = 0
        model.fit(x, **fit_kwargs)

        for _ in range(self._control_horizons):
            control = model.act_wrapper(x, **act_kwargs)
            cost = system.calc_control_cost_wrapper(control)
            total_cost += cost
            assert np.all(cost <= max_control_cost), "Control cost exceeded!"
            x = system.make_data_wrapper(
                init_conds=x[:, -1], control=control, timesteps=timesteps, noisy=noisy)
            model.fit(x, **fit_kwargs)

        return total_cost

    @staticmethod
    def _append_result(result, rep_id, n, latent_dim, embed_dim, timesteps, total_cost , error, ood_error=None):
        result['rep'].append(rep_id)
        result['n'].append(n)
        result['latent_dim'].append(latent_dim)
        result['embed_dim'].append(embed_dim)
        result['timesteps'].append(timesteps)
        result['error'].append(error)
        result['total_cost'].append(total_cost)
        result['ood_error'].append(ood_error)
    
    @staticmethod
    def _init_result_dict():
        result = {k: [] for k in [
            "rep", 
            "latent_dim", 
            "embed_dim", 
            "timesteps",
            "n", 
            "error", 
            "ood_error",
            "total_cost"]}
        return result

    def system_run(self, 
                    rep_id, 
                    latent_dim, 
                    embed_dim, 
                    model_cls : type[AbstractModel],
                    model_kwargs : dict = None,
                    fit_kwargs : dict = None,
                    act_kwargs : dict = None,
                    noisy : bool = False,
                    test_ood : bool = False,
                    system_seed=None, 
                    model_seed=None,
                    **kwargs
                    ):
        """
        For a given system latent dimension and embedding dimension, instantiates system and evaluates reps of
        iterating over the number of trajectories N.

        Note that model seed in model_kwargs and system_seed in system_kwargs takes precedence over the seed passed to this function.
        """
        result = Challenge._init_result_dict()
        
        if embed_dim < latent_dim:
            return

        # Seed in system_kwargs takes precedence over the seed passed to this function.
        system = self._system_cls(latent_dim, embed_dim, **{"seed":system_seed, **self._system_kwargs})
        
        # Create data
        largest_N = max(self._N)
        training_set = self._gen_trainset(system, largest_N, self._t, noisy)
        test_set = self._gen_testset(system, in_dist=True)
        ood_test_set = self._gen_testset(system, in_dist=False)
        
        max_control_cost = self._max_control_cost_per_dim * latent_dim

        # On each subset of the training set, we retrain the model from scratch (initialized with the same random seed). If you don't, then # of training epochs will scale with N. This would confound the effect of training set size with training time.
        for n in self._N:
            # Create Model. Seed in model_kwargs takes precedence over the seed passed to this function.
            model = model_cls(embed_dim, self._t, max_control_cost, **{"seed": model_seed, **model_kwargs})
            training_set_n = training_set[:n] #train on subset of training set
            total_cost = self._fit_model(system, model, training_set_n, self._t, max_control_cost, fit_kwargs, act_kwargs, noisy)
            pred = model.predict_wrapper(test_set[:, 0], self._test_timesteps)
            error = system.calc_error_wrapper(pred, test_set)
            ood_error = None
            if test_ood:
                ood_pred = model.predict_wrapper(ood_test_set[:, 0], self._test_timesteps)
                ood_error = system.calc_error_wrapper(ood_pred, ood_test_set)
            #TODO: fix logging? Should we use a logger?
            print(f"{id=}, {n=}, {latent_dim=}, {embed_dim=}, t={self._t}, control_h={self._control_horizons}, {rep_id=}, {total_cost}, {error=:0.3}, {ood_error=:0.3},model_seed={model._seed}, sys_seed={system._seed}")
            Challenge._append_result(result, rep_id, n, latent_dim, embed_dim, self._t, total_cost, error, ood_error)

        data =pd.DataFrame(result)
        data['system_seed'] = system_seed
        data['model_seed'] = model_seed
        return data