import numpy as np


class Latent(object):
    def __init__(self):
        pass

    def sample(self, batch_size):
        raise NotImplementedError


class UniformLatent(Latent):
    def __init__(self, dim, low=-1.0, high=1.0,
                 *args, **kwargs):
        super(UniformLatent, self).__init__(*args, **kwargs)
        self.dim = dim
        self.low = low
        self.high = high

    def sample(self, batch_size):
        return np.random.uniform(
            self.low, self.high, size=(batch_size, self.dim))


class GaussianLatent(Latent):
    def __init__(self, dim, loc=0.0, scale=1.0,
                 *args, **kwargs):
        super(GaussianLatent, self).__init__(*args, **kwargs)
        self.dim = dim
        self.loc = loc
        self.scale = scale

    def sample(self, batch_size):
        return np.random.normal(
            loc=self.loc, scale=self.scale, size=(batch_size, self.dim))
