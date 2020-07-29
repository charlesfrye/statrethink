# TODO: convert to logs
# TODO: review use of numpy arrays
# TODO: posterior into prior
# TODO: reconsider interface for prior/likelihood setting

from abc import abstractmethod
import base64
from io import BytesIO

import autograd.numpy as np
import matplotlib


class GridGolem(object):

    def __init__(self, grid_spacing=0.001):
        self.grid = self.make_grid(grid_spacing)
        self._posterior = self.prior

    def update(self, observations):
        self._posterior = self.update_posterior(observations)

    def update_posterior(self, observations):

        N = len(self.grid)
        normalization_term = 1 / N * np.sum(
            self._unnormalized_posterior(observations, self.grid))

        def posterior(p):
            return np.squeeze(
                self._unnormalized_posterior(observations, p) / normalization_term)

        return posterior

    def _unnormalized_posterior(self, observations, ps):
        ps = np.atleast_1d(ps)
        observations = np.atleast_1d(observations)

        likelihood_term = self.likelihood(observations, ps)
        prior_term = self.prior(ps)

        return likelihood_term * prior_term

    def __call__(self, ps):
        return self._posterior(ps)

    def _repr_html_(self):
        # Generate the figure **without using pyplot**.
        fig = matplotlib.figure.Figure()
        ax = fig.subplots()
        ax.plot(self.grid, self._posterior(self.grid), lw=4, c="w")

        # Save it to a temporary buffer.
        buf = BytesIO()
        fig.savefig(buf, format="png")
        # Embed the result in the html output.
        data = base64.b64encode(buf.getbuffer()).decode("ascii")

        return f"<img src='data:image/png;base64,{data}'/>"

    def make_grid(self, grid_spacing):
        return np.arange(0, 1 + grid_spacing, grid_spacing)

    @abstractmethod
    def likelihood(observations, ps):
        raise NotImplementedError

    @abstractmethod
    def prior(ps):
        raise NotImplementedError
