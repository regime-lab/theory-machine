from sklearn.mixture import BayesianGaussianMixture
import numpy as np
import scipy 

class DGPTheory:
    
    """
    A class that represents a Theory, which is a simple wrapper for three variables: array of means,
    array of covariance matrices, and transition probability matrix for each time step. 
    """
    def __init__(self, means, covariances, transition_kernel):
        """
        Constructor that accepts arrays of means, covariance matrices, and a time-varying transition 
        probability matrix.

        Parameters:
        - means (array): An array of means.
        - covariances (array): An array of covariance matrices.
        - transition_kernel (3d array): A transition probability matrix for each time step.  
        """
        self.covariances = covariances
        self.transition_kernel = transition_kernel   
        self.means = means

    """
    Sample from the theory as a data generating process.
    """
    def sample(self, num_samples=100): 
        
        # Declare empty states
        states = np.zeros(num_samples)

        # Sample the posterior distribution states based on transition matrix and multivariate normal 
        if len(self.covariances) > 1: 
            states = [ np.random.choice(len(self.covariances), 
                                        p=self.transition_kernel[i][int(states[i-1])]) for i in range(num_samples) ]

        # Get X, and y dim 
        Xy = np.array([ 
                       np.random.multivariate_normal(
                           mean=self.means[state], 
                           cov=self.covariances[state]) for state in map(int, states) ])
        return Xy, states
