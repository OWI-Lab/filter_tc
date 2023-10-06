import numpy as np

class CustomGammaDistribution:
    def __init__(self, shape, scale, loc=0, num_terms=100, epsilon=1e-10):
        self.shape = shape
        self.scale = scale
        self.loc = loc
        self.num_terms = num_terms  # Number of terms in the series
        self.epsilon = epsilon  # Small constant to avoid division by zero

    def pdf(self, x):
        x_shifted = x - self.loc

        term1 = (1 / (self.scale ** self.shape * np.math.gamma(self.shape)))
        term2 = x_shifted ** (self.shape - 1)
        term3 = np.exp(-x_shifted / self.scale)
        
        # Calculate the gamma function using scipy for precision
        gamma_shape = np.math.gamma(self.shape)
        
        # Use a loop to approximate the series with more terms
        series_sum = 0
        for k in range(self.num_terms):
            numerator = (-1) ** k * (x_shifted ** k)
            denominator = (self.shape + k) * gamma_shape + self.epsilon  # Add epsilon to avoid division by zero
            series_term = numerator / denominator
            series_sum += series_term
        
        return term1 * term2 * term3 * series_sum
