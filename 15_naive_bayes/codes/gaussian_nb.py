from multinomial_nb import MultinomialNB
import numpy as np

class GaussianNB(MultinomialNB):
    def cal_conditional_prob(self, sample_f):
        mu = np.mean(sample_f)
        sigma = np.std(sample_f)
        return (mu, sigma)

    def cal_gaussian(self, mu, sigma, x):
        return ( 1.0/(sigma * np.sqrt(2 * np.pi)) *
                        np.exp( - (x - mu)**2 / (2 * sigma**2)) )

    def predict_ij(self, mu_sigma, j):
        return self.cal_gaussian(mu_sigma[0], mu_sigma[1], j)