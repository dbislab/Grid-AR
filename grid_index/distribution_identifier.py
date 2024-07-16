from abc import ABC, abstractmethod
from scipy import stats

class MyDistribution(ABC):
    def __init__(self, dist_data):
        pass

    @property
    @abstractmethod
    def distribution_params(self):
        pass

    @property
    @abstractmethod
    def dist_name(self):
        pass

    @abstractmethod
    def sample_data(self, num_samples):
        pass

    @abstractmethod
    def set_dist_name(self):
        pass

class CustomScipyDist(MyDistribution):

    def __init__(self, dist_data):
        self._distribution_params = dict()
        self._distribution_params[0] = dist_data[0]
        self._distribution_params[1] = dist_data[1]

    @property
    def distribution_params(self):
        return self._distribution_params

    def sample_data(self, num_samples):
        dist = getattr(stats, self._distribution_params[0])
        return dist.rvs(*self._distribution_params[1], size=num_samples)