import numpy as np
from scipy.stats import beta, binom, norm, uniform
from tqdm import tqdm


class BetaBinomial:
    def __init__(self, beta_a_param=2, beta_b_param=2, n_heads=8, n_total=30, initial_probability_heads=.5,
                 n_iter=5000):
        self.beta_a_param = beta_a_param
        self.beta_b_param = beta_b_param
        self.n_heads = n_heads
        self.n_total = n_total
        self.initial_probability_heads = initial_probability_heads
        self.n_iter = n_iter
        self.probability_heads_traces = self._initialize_probability_traces(initial_probability_heads)
        self.posteriors = self._initialize_posteriors(initial_probability_heads)
        self.proposals = np.zeros(self.n_iter - 1)
        self.accept_array = np.zeros(self.n_iter - 1, dtype=bool)

    def _calculate_prior_density_p_heads(self, probability_heads):
        return beta.pdf(a=self.beta_a_param, b=self.beta_b_param, x=probability_heads)

    def _calculate_likelihood_density_n_heads_given_p(self, probability_heads):
        if (probability_heads < 0) or (probability_heads > 1):
            return 0
        else:
            return binom.pmf(p=probability_heads, n=self.n_total, k=self.n_heads)

    def calculate_posterior(self, probability_heads):
        prior_density = self._calculate_prior_density_p_heads(probability_heads)
        likelihood_density = self._calculate_likelihood_density_n_heads_given_p(probability_heads)
        return prior_density * likelihood_density

    def _initialize_probability_traces(self, initial_probability_heads):
        probability_traces = np.zeros(self.n_iter)
        probability_traces[0] = initial_probability_heads
        return probability_traces

    def _initialize_posteriors(self, initial_probability_heads):
        posteriors = np.zeros(self.n_iter)
        posteriors[0] = self.calculate_posterior(initial_probability_heads)
        return posteriors

    @staticmethod
    def _accept_or_reject_proposal(new_posterior, old_posterior):
        ratio = new_posterior / old_posterior
        if ratio > uniform.rvs():
            return True
        else:
            return False

    def fit(self, proposal_std=.2):
        for it in tqdm(range(self.n_iter - 1)):
            self.proposals[it] = norm.rvs(loc=self.probability_heads_traces[it], scale=proposal_std)
            proposal_posterior = self.calculate_posterior(self.proposals[it])
            accept = self._accept_or_reject_proposal(proposal_posterior, self.posteriors[it])
            if accept:
                self.posteriors[it + 1] = proposal_posterior
                self.probability_heads_traces[it + 1] = self.proposals[it]
                self.accept_array[it] = accept
            else:
                self.posteriors[it + 1] = self.posteriors[it]
                self.probability_heads_traces[it + 1] = self.probability_heads_traces[it]
