# -*- coding: utf-8 -*-

import numpy as np
from sklearn.utils import check_random_state
from skopt.space.transformers import Pipeline, Identity, Normalize
from skopt.sampler.lhs import Lhs


def _transpose_list_array(x):
    """Transposes a list matrix

    From skopt.
    """

    n_dims = len(x)
    assert n_dims > 0
    n_samples = len(x[0])
    rows = [None] * n_samples
    for i in range(n_samples):
        r = [None] * n_dims
        for j in range(n_dims):
            r[j] = x[j][i]
        rows[i] = r
    return rows


def normalize(X, bounds):
    """
    Normalize points in X.

    Keyword Arguments:
    X -- Array of points to normalize
    bounds -- Definition space of points in X

    Returns:
    Xt -- Array of normalized points
    """
    # Pack by dimension
    columns = []
    for dim in range(len(bounds)):
        columns.append([])
    for i in range(len(X)):
        for j in range(len(bounds)):
            columns[j].append(X[i][j])

    # Normalize
    for j,bounds in enumerate(bounds):
        transformer = Pipeline([Identity(), Normalize(bounds[0], bounds[1])])
        columns[j] = transformer.transform(columns[j])

    # Repack as an array
    Xt = np.hstack([np.asarray(c).reshape((len(X), -1)) for c in columns])

    return Xt


class TargetSpace(object):
    """
    Holds the param-space coordinates (X) and target values (Y)
    """

    def __init__(self, target_function, NObj, pbounds, constraints,
                 RandomSeed, init_points=2, verbose=False):
        """
        Keyword Arguments:
        target_function -- list of Functions to be maximized
        NObj            -- Number of objective functions
        pbounds         -- numpy array with bounds for each parameter
        """

        super(TargetSpace, self).__init__()

        self.vprint = print if verbose else lambda *a, **k: None

        self.RS = check_random_state(RandomSeed)

        self.target_function = target_function
        self.NObj = NObj

        self.ParetoSize = 0 # the nb of non-dominated points that have been evaluated

        self.pbounds = pbounds
        self.constraints = constraints
        self.init_points = init_points

        if len(self.constraints) == 0:
            self.__NoConstraint = True

        # Find number of parameters
        self.NParam = self.pbounds.shape[0]

        self.q = self.NParam
        # Number of observations
        self._NObs = 0

        self._X = None  # points at which target func has been evaluated
        self._Y = None  # corresponding dominance (1 if the point is non-dominated, 0 otherwise)
        self._W = None
        self._F = None  # corresponding obj funcs values
        self.length = 0 # Nb of observations stored

        return

    # % Satisfy constraints
    def SatisfyConstraints(self, x, tol=1.0e-6):
        for cons in self.constraints:
            y = cons['fun'](x)
            if cons['type'] == 'eq':
                if np.abs(y) > tol:
                    return False
            elif cons['type'] == 'ineq':
                if y < -tol:
                    return False
        return True

    # % Other
    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    @property
    def W(self):
        return self._W

    @property
    def x(self):
        return self.X[:self.length]

    @property
    def y(self):
        return self.Y[:self.length]

    @property
    def f(self):
        return self._F[:self.length]

    @property
    def w(self):
        return self.W[:self.length]

    @property
    def _n_alloc_rows(self):
        """ Number of allocated rows """
        return 0 if self._X is None else self._X.shape[0]

    def __len__(self):
        return self.length

    # % CONTAINS
    def __contains__(self, x):
        try:
            return x in self._X
        except:                 # noqa
            return False

    def __repr__(self):
        HeaderX = ''.join(f'   X{i}    ' for i in range(self.NParam))
        HeaderY = ''.join(f'   F{i}    ' for i in range(self.NObj))
        Out = HeaderX+' | '+HeaderY+'\n'
        for i in range(self.length):
            LineX = ''.join(f'{i:+3.1e} ' for i in self.x[i])
            LineY = ''.join(f'{i:+3.1e} ' for i in self.f[i])
            Out += LineX+' | '+LineY+'\n'
        return Out

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.x[key], self.f[key]
        elif isinstance(key, int):
            if key < self.length:
                return self.x[key], self.f[key]
            else:
                raise KeyError(f"key ({key}) larger than {self.length}")
        else:
            raise TypeError(f"Invalid key type for space")

    # % TRANSFORM
    def normalize(self, X):
        """
        Normalize points in X.

        Keyword Arguments:
        X -- Array of points to normalize

        Returns:
        Xt -- Array of normalized points
        """
        # Pack by dimension
        columns = []
        for dim in range(self.NParam):
            columns.append([])
        for i in range(len(X)):
            for j in range(self.NParam):
                columns[j].append(X[i][j])

        # Normalize
        for j,bounds in enumerate(self.pbounds):
            transformer = Pipeline([Identity(), Normalize(bounds[0], bounds[1])])
            columns[j] = transformer.transform(columns[j])

        # Repack as an array
        Xt = np.hstack([np.asarray(c).reshape((len(X), -1)) for c in columns])

        return Xt


    def inverse_normalize(self, Xt):
        """Inverse normalize points in Xt.

        Keyword Arguments:
        Xt -- Array of points to inverse normalize

        Returns:
        X -- Array of inverse normalized points
        """
        # Proceed one dimension at a time
        columns = []
        Xt = np.asarray(self.NParam)
        for j, bounds in enumerate(self.pbounds):
            transformer = Pipeline([Identity(), Normalize(bounds[0], bounds[1])])
            inv_transform = transformer.inverse_transform(Xt[:, j])
            # Deal with var format and types
            if isinstance(inv_transform, list):
                inv_transform = np.asarray(inv_transform)
            inv_transform = np.clip(inv_transform, bounds[0], bounds[1]).astype(float)
            # necessary, otherwise the type is converted to a numpy type
            inv_transform = getattr(inv_transform, "tolist", lambda: value)()
            columns.append(inv_transform)

        # Transpose
        X = _transpose_list_array(columns)

        return X


    # % RANDOM POINTS
    def random_points(self, num):
        """
        Creates random points within the bounds of the space

        Keyword Arguments:
        num  -- number of random points to create

        Returns:
        data -- [num x NParam] array of points
        """

        # Latin Hypercube Sampling
        lhs_sampler = Lhs()
        samples = lhs_sampler.generate(self.pbounds, num, random_state=self.RS.randint(0, np.iinfo(np.int32).max))
        return samples
        #data = np.empty((num, self.NParam))

        #for i in enumerate(data):
        #    while True:
        #        DD = self.OneRandomPoint(self.NParam, self.pbounds, self.RS)
        #        if self.SatisfyConstraints(DD):
        #            data[i[0]] = DD
        #            break
        #return data.tolist()

    # % return one point
    #@staticmethod
    #def OneRandomPoint(NParam, pbounds, RS):
    #    x = np.empty(NParam)
    #    counter = 0
    #    for b in pbounds:
    #        if b[0] is None and b[1] is None:
    #            x[counter] = RS.normal(size=1)
    #        elif b[0] is None:
    #            x[counter] = b[1] - RS.exponential(size=1)
    #        elif b[1] is None:
    #            x[counter] = b[0] + RS.exponential(size=1)
    #        else:
    #            x[counter] = RS.uniform(low=b[0], high=b[1], size=1)
    #        counter += 1
    #    return x.tolist()

    # % OBSERVE POINT
    def observe_point(self, x):
        """
        Evaluates a single point x, to obtain the value y and them records
        them as observations

        NOTE: if x has been previously seen returns a cached value of y

        Keyword Arguments:
        x    -- a single point, w/ len(x) == self.NParam
        """

        assert x.size == self.NParam, 'x must have the same dimension'

        f = self.target_function(x)

        try:
            NewF = []
            for ff in f:
                NewF.append(ff[0])
            f = NewF
        except:                 # noqa
            pass

        self.add_observation(x, f)

        return f

    # % ADD POINT
    def add_observation(self, x, f):
        """
        Append a point and its target values to the known data

        Keyword Arguments:
        x    -- a single point
        y    --  target function value
        """

        assert x.size == self.NParam, 'x must have the same dimension'

        if self.length >= self._n_alloc_rows:
            self._allocate((self.length + 1) * 2) # increase the arrays to store new observation and future ones

        self._X[self.length] = x
        self._F[self.length] = f
        self._Y[self.length] = self.dominated(f)

        self.length += 1
        self.UpdateDominance()
        # self.DominanceWeight()

        self.ParetoSize = len(np.where(self._Y == 1)[0])

        return

    # % Pareto Set
    def ParetoSet(self):
        iPareto = np.where(self._Y == 1)[0]
        return self._F[iPareto], self._X[iPareto]

    # % Update dominance
    def UpdateDominance(self):
        for i in range(self.length-1):
            if self._Y[i] == 1:
                if self.Larger(self._F[self.length-1], self._F[i]):
                    self._Y[i] = 0 # the new point dominates point i so i is no longer non-dominated
        return

    def DominanceWeight(self):
        for i in range(self.length):
            self.w[i] = 0
            for j in range(self.length):
                if i != j:
                    if self.Larger(self._F[j], self._F[i]):
                        self.w[i] += 1
            self.w[i] = np.exp(-self.w[i])
        return

    # % test for dominance
    def dominated(self, f):
        # returns 1 if non-dominated 0 if dominated
        Dominated = 1
        for i in range(self.length):
            if self._Y[i] == 1: # point i non-dominated
                if self.Larger(self._F[i], f): # point i dominates f
                    Dominated = 0
                    break
        return Dominated

    # % Compare two lists
    @staticmethod
    def Larger(X, Y):
        # test if X > Y (in the dominance sense)
        Dominates = True
        NumberOfLarger = 0
        for i, x in enumerate(X): # for each objective
            if x > Y[i]:
                Dominates = Dominates and True
                NumberOfLarger += 1
            elif x == Y[i]:
                Dominates = Dominates and True
            else:
                Dominates = Dominates and False
                break
        Dominates = Dominates and (NumberOfLarger > 0)
        return Dominates

    # % allocate memory

    def _allocate(self, num):
        """
        Keyword Arguments:
        num  -- number of points to be allocated
        """

        if num <= self._n_alloc_rows:
            raise ValueError('num must be larger than current array length')

        #  Allocate new memory
        _Xnew = np.empty((num, self.NParam))
        _Ynew = np.empty(num, dtype=object)
        _Wnew = np.empty(num, dtype=object)
        _Fnew = np.empty((num, self.NObj))
        # _Fnew = np.empty(num,dtype=object)
        # Copy the old data into the new
        if self._X is not None:
            _Xnew[:self.length] = self._X[:self.length]
            _Ynew[:self.length] = self._Y[:self.length]
            _Wnew[:self.length] = self._W[:self.length]
            _Fnew[:self.length] = self._F[:self.length]

        self._X = _Xnew
        self._Y = _Ynew
        self._W = _Wnew
        self._F = _Fnew

        return

    # % write relevant information to file
    def WriteSpace(self, filename="space"):

        Info = [self.NObj,
                self.NParam,
                self._NObs,
                self.length]

        np.savez(filename,
                 X=self._X,
                 Y=self._Y,
                 F=self._F,
                 I=Info)        # noqa

        return

    # % read relevant information from file
    def ReadSpace(self, filename="space"):

        Data = np.load(filename)

        self.NObj = Data["I"][0]
        self.NParam = Data["I"][1]
        self._NObs = Data["I"][2]
        self.length = Data["I"][3]

        self._allocate((self.length + 1) * 2)

        self._X = Data["X"]
        self._Y = Data["Y"]
        self._F = Data["F"]

        return
