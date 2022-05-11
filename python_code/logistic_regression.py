import numpy as np
from scipy.stats import bernoulli, norm, uniform
from tqdm import tqdm


class BayesianLogisticRegression:
    def __init__(self, X, y, n_iter, betas_initial, prior_mean_betas=0, prior_std_betas=1):
        self.X = X
        self.y = y
        self.n_iter = n_iter
        self.prior_mean_betas = prior_mean_betas
        self.prior_std_betas = prior_std_betas
        self.n_betas = self.X.shape[1]
        self.betas_traces = self._initialize_betas_traces(betas_initial)
        self.posteriors = self._initialize_betas_posteriors(betas_initial)
        self.accept_array = np.zeros(self.n_iter - 1, dtype=bool)
        self.proposals = np.zeros((self.n_iter - 1, self.n_betas))

    def _calculate_logit(self, betas):
        return 1 / (1 + np.exp(-self.X @ betas))

    def _calculate_prior_density_betas(self, betas):
        betas_priors = norm.pdf(loc=self.prior_mean_betas, scale=self.prior_std_betas, x=betas)
        return np.prod(betas_priors)

    def _calculate_likelihood_density_y(self, logits):
        return np.prod([bernoulli.pmf(p=logits[n], k=self.y[n]) for n in range(self.y.shape[0])])

    def _initialize_betas_traces(self, betas_initial):
        betas_traces = np.zeros((self.n_iter, self.n_betas))
        betas_traces[0, :] = betas_initial
        return betas_traces

    def _initialize_betas_posteriors(self, initial_betas):
        posteriors = np.zeros(self.n_iter)
        initial_posterior = self.calculate_posterior(initial_betas)
        posteriors[0] = initial_posterior
        return posteriors

    def calculate_posterior(self, betas):
        prior_density = self._calculate_prior_density_betas(betas)
        logits = self._calculate_logit(betas)
        likelihood_density = self._calculate_likelihood_density_y(logits)
        return likelihood_density * prior_density

    @staticmethod
    def _accept_or_reject_proposal(new_posterior, old_posterior):
        ratio = new_posterior / old_posterior
        if ratio > uniform.rvs():
            return True
        else:
            return False

    def fit(self, proposal_std=.2):
        for it in tqdm(range(self.n_iter - 1)):
            self.proposals[it, :] = norm.rvs(loc=self.betas_traces[it, :], scale=np.ones(self.n_betas) * proposal_std)
            betas_proposal_posterior = self.calculate_posterior(self.proposals[it, :])
            accept = self._accept_or_reject_proposal(betas_proposal_posterior, self.posteriors[it])
            if accept:
                self.posteriors[it + 1] = betas_proposal_posterior
                self.betas_traces[it + 1, :] = self.proposals[it]
                self.accept_array[it] = True
            else:
                self.posteriors[it + 1] = self.posteriors[it]
                self.betas_traces[it + 1, :] = self.betas_traces[it, :]
