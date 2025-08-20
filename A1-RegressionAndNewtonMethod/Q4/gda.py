# Imports - you can add any other permitted libraries
import numpy as np
from numpy.linalg import inv, det, LinAlgError
import matplotlib.pyplot as plt

# You may add any other functions to make your code more modular. However,
# do not change the function signatures (name and arguments) of the given functions,
# as these functions will be called by the autograder.

class GaussianDiscriminantAnalysis:
    # Assume Binary Classification (hard-coded for binary classification)
    # Also, here I'm assuming that you'll be running the algorithm separately for same covariance and different covariance
    def __init__(self):
        # Normilization parameters
        self.mean = None
        self.std = None
        # Parameter for Bernoulli distribution
        self.phi = None 
        self.mu_0 = None
        self.mu_1 = None
        # Parameters for Gaussian distribution
        """ Initialize the model for same covariance """
        self.sigma_s = None
        """ Iniatialize the model for different covariance """
        self.sigma_0_d = None
        self.sigma_1_d = None

    def _normalize_features(self, X):   
        # Compute mean and standard deviation of features
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        
        # just in case std is zero
        self.std[self.std < 1e-10] = 1e-10

        X_norm = (X - self.mean) / (self.std)
        return X_norm
    
    def fit(self, X, y, assume_same_covariance=False):

        X = self._normalize_features(X)
        m = X.shape[0]
        X_0 = X[y == 0]
        X_1 = X[y == 1]

        """calculate parameters"""
        # calculate phi
        self.phi = np.mean(y)

        # calculate mu_0 and mu_1
        mu_0 = np.mean(X_0, axis=0)
        mu_1 = np.mean(X_1, axis=0)

        # Note: In the calculations below, we are assuming the returned sigmas to +ve semi definite (and invertible)
        if assume_same_covariance:
            sigma = np.zeros((X.shape[1], X.shape[1]))
            for i in range(len(X_0)):
                diff = (X_0[i] - mu_0).reshape(-1, 1)
                sigma += diff @ diff.T
                
            for i in range(len(X_1)):
                diff = (X_1[i] - mu_1).reshape(-1, 1)
                sigma += diff @ diff.T
                
            sigma = sigma / m

            self.mu_0_s = mu_0
            self.mu_1_s = mu_1
            self.sigma_s = sigma

            self.sigma_0_d = None
            self.sigma_1_d = None
                
            return mu_0, mu_1, sigma 
        
        else:
            # Calculate sigma_0
            sigma_0 = np.zeros((X.shape[1], X.shape[1]))
            for i in range(len(X_0)):
                diff = (X_0[i] - mu_0).reshape(-1, 1)
                sigma_0 += diff @ diff.T
            sigma_0 = sigma_0 / len(X_0)

            # Calculate sigma_1
            sigma_1 = np.zeros((X.shape[1], X.shape[1]))
            for i in range(len(X_1)):
                diff = (X_1[i] - mu_1).reshape(-1, 1)
                sigma_1 += diff @ diff.T
            sigma_1 = sigma_1 / len(X_1)   

            self.mu_0_d = mu_0
            self.mu_1_d = mu_1
            self.sigma_0_d = sigma_0
            self.sigma_1_d = sigma_1

            self.sigma_s = None
            
            return mu_0, mu_1, sigma_0, sigma_1  

    def predict_s(self, X):
        """
        Predict using the shared covariance model (vectorized).
        """
        X = self._normalize_features(X)
        try:
            sigma_inv = inv(self.sigma_s)
            
            # Precompute terms that don't depend on x
            const_0 = np.log(1 - self.phi)
            const_1 = np.log(self.phi)
            
            # Vectorized calculation of discriminant scores
            diff_0 = X - self.mu_0_s  
            diff_1 = X - self.mu_1_s  

            score_0 = -0.5 * np.sum(diff_0 @ sigma_inv * diff_0, axis=1) + const_0 
            score_1 = -0.5 * np.sum(diff_1 @ sigma_inv * diff_1, axis=1) + const_1 

            y_pred = (score_1 > score_0).astype(int) # Vectorized comparison and type conversion
            
        except LinAlgError:
            raise ValueError("Singular covariance matrix in shared covariance model")
            
        return y_pred


    def predict_d(self, X):
        """
        Predict using the different covariance model (vectorized).
        """
        X = self._normalize_features(X)
        try:
            sigma_0_inv = inv(self.sigma_0_d)
            sigma_1_inv = inv(self.sigma_1_d)
            
            # Precompute terms that don't depend on x
            const_0 = -0.5 * np.log(det(self.sigma_0_d)) + np.log(1 - self.phi)
            const_1 = -0.5 * np.log(det(self.sigma_1_d)) + np.log(self.phi)
            
            # Vectorized calculation of discriminant scores
            diff_0 = X - self.mu_0_d  
            diff_1 = X - self.mu_1_d  

            score_0 = -0.5 * np.sum(diff_0 @ sigma_0_inv * diff_0, axis=1) + const_0
            score_1 = -0.5 * np.sum(diff_1 @ sigma_1_inv * diff_1, axis=1) + const_1

            y_pred = (score_1 > score_0).astype(int)

        except LinAlgError:
            raise ValueError("Singular covariance matrix in different covariance model")
            
        return y_pred
    
    def predict(self, X):
        if self.sigma_s is not None:
            return self.predict_s(X)
        else:
            return self.predict_d(X)
        
    # These boundary equations we obtain are for the normalized features
    def db_s(self):
        """
        Return function for shared covariance decision boundary.
        The boundary is where the discriminant functions are equal:
        δ₁(x) = δ₀(x)
        """
        sigma_inv = inv(self.sigma_s)
        diff_mu = self.mu_1_s - self.mu_0_s
        
        # Linear decision boundary coefficients
        w = sigma_inv @ diff_mu  # direction vector
        b = -0.5 * (self.mu_1_s.T @ sigma_inv @ self.mu_1_s - 
                    self.mu_0_s.T @ sigma_inv @ self.mu_0_s) + np.log(self.phi / (1 - self.phi))
        
        # Create a string representation of the function
        equation = f"{w[0]:.3f}x₁ + {w[1]:.3f}x₂ + {b:.3f} = 0"
                
        # Return both the decision function and the coefficients for plotting
        return {
            'type': 'linear',
            'w': w,
            'b': b,
            'equation': equation,
            'decision_function': lambda x: x @ w + b
        }

    def db_d(self):
        """
        Return function for different covariance decision boundary.
        The boundary is quadratic due to different covariance matrices.
        """
        sigma_0_inv = inv(self.sigma_0_d)
        sigma_1_inv = inv(self.sigma_1_d)
        
        # Quadratic decision boundary function
        def quad_boundary(x):
            x = (x - self.mean) / self.std  # normalize
            diff_0 = x - self.mu_0_d
            diff_1 = x - self.mu_1_d
            
            return (diff_0.T @ sigma_0_inv @ diff_0 - 
                   diff_1.T @ sigma_1_inv @ diff_1 + 
                   np.log(det(self.sigma_0_d) / det(self.sigma_1_d)) + 
                   2 * np.log(self.phi / (1 - self.phi)))
        
        # Return both the decision function and the matrices for plotting
        return {
            'type': 'quadratic',
            'decision_function': quad_boundary
        }

    def decision_boundary(self):
        if self.sigma_s is not None:
            return self.db_s()
        else:
            return self.db_d()
        
    def plot_linear_decision_boundary(self, X, y):
        plt.figure(figsize=(8, 6))
        X = self._normalize_features(X)

        # Scatter plot of data
        plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='o', color='red', label='Alaska')
        plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='^', color='blue', label='Canada')

        boundary = self.decision_boundary()

        if boundary['type'] == 'linear':
            w = boundary['w']
            b = boundary['b']
            x_vals = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
            y_vals = (-b - w[0] * x_vals) / w[1]  # Solve for x2
            plt.plot(x_vals, y_vals, 'k-', label='Decision Boundary')

            plt.xlabel("x1 (normalized)")
            plt.ylabel("x2 (normalized)")
            plt.legend()
            plt.title("GDA Linear Decision Boundary (normalised)")
            plt.show()

    def quadratic_boundary_equation(self):
        """
        Returns a human-readable equation of the quadratic decision boundary.
        """
        # Compute inverses of covariance matrices
        sigma_0_inv = inv(self.sigma_0_d)
        sigma_1_inv = inv(self.sigma_1_d)
        
        # Compute quadratic term (A matrix)
        A = 0.5 * (sigma_0_inv - sigma_1_inv)

        # Compute linear term (B vector)
        B = self.mu_1_d.T @ sigma_1_inv - self.mu_0_d.T @ sigma_0_inv

        # Compute constant term (C)
        C = (0.5 * (self.mu_0_d.T @ sigma_0_inv @ self.mu_0_d - 
                    self.mu_1_d.T @ sigma_1_inv @ self.mu_1_d) + 
            0.5 * np.log(det(self.sigma_0_d) / det(self.sigma_1_d)) + 
            np.log((1 - self.phi) / self.phi))

        # Human-readable equation
        eq_str = f"{A[0,0]:.3f} x₁² + {A[1,1]:.3f} x₂² + {2*A[0,1]:.3f} x₁x₂ + {B[0]:.3f} x₁ + {B[1]:.3f} x₂ + {C:.3f} = 0"

        return eq_str
    

    def plot_quadratic_decision_boundary(self, X, y):
        plt.figure(figsize=(8, 6))
        X = self._normalize_features(X)

        plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='o', color='red', label='Alaska')
        plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='^', color='blue', label='Canada')

        sigma_0_inv = inv(self.sigma_0_d)
        sigma_1_inv = inv(self.sigma_1_d)

        A = 0.5 * (sigma_0_inv - sigma_1_inv)
        B = (self.mu_1_d.T @ sigma_1_inv - self.mu_0_d.T @ sigma_0_inv)
        C = (0.5 * (self.mu_0_d.T @ sigma_0_inv @ self.mu_0_d -
                    self.mu_1_d.T @ sigma_1_inv @ self.mu_1_d) +
            0.5 * np.log(det(self.sigma_0_d) / det(self.sigma_1_d)) +
            np.log((1 - self.phi) / self.phi))

        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.01),
                            np.arange(x2_min, x2_max, 0.01))
        Z = np.zeros_like(xx1)

        xx = np.c_[xx1.ravel(), xx2.ravel()]
        Z = A[0, 0] * xx[:, 0]**2 + 2 * A[0, 1] * xx[:, 0] * xx[:, 1] + A[1, 1] * xx[:, 1]**2 + B[0] * xx[:, 0] + B[1] * xx[:, 1] + C
        Z = Z.reshape(xx1.shape)

        contour = plt.contour(xx1, xx2, Z, [0], colors='k', linestyles='solid')
        plt.plot([], [], 'k', label='Quad Boundary')  # Manually adding to legend

        plt.xlabel("x1 (normalized)")
        plt.ylabel("x2 (normalized)")
        plt.legend()  # Now the label will be picked up
        plt.title("GDA Quadratic Decision Boundary (normalised)")
        plt.show()

