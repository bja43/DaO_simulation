import numpy as np

from numpy.random import default_rng


rng = default_rng()

err = lambda *x: rng.normal(0, np.sqrt(x[0]), x[1])
err = lambda *x: rng.gumbel(0, np.sqrt(6.0) * np.sqrt(x[0]) / np.pi, x[1])
err = lambda *x: rng.exponential(np.sqrt(x[0]), x[1])
err = lambda *x: rng.laplace(0, np.sqrt(x[0]) / np.sqrt(2), x[1])
err = lambda *x: rng.uniform(-np.sqrt(3) * np.sqrt(x[0]), np.sqrt(3) * np.sqrt(x[0]), x[1])
