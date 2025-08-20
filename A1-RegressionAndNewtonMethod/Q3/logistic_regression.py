# Imports - you can add any other permitted libraries
import numpy as np

# You may add any other functions to make your code more modular. However,
# do not change the function signatures (name and arguments) of the given functions,
# as these functions will be called by the autograder.

class LogisticRegressor:
    def __init__(self):
        # parameters - saves the state of the model
        self.theta = None

        # normalization parameters
        self.mean = None
        self.std = None

        # hyperparameters eps and n_iter : for convergence (eps/n_iter)
        self.eps = 1e-5
        self.n_iter = 1000
        
    def _normalize_features(self, X):   
        # Compute mean and standard deviation of features
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        
        # just in case std is zero
        self.std[self.std < 1e-10] = 1e-10

        X_norm = (X - self.mean) / (self.std)
        return X_norm
    
    def _sigmoid(self, z):  
        z = np.clip(z, -500, 500)  # Prevent overflow
        return 1 / (1 + np.exp(-z))

    def _compute_gradient(self, X, y, theta):
        m = len(y)
        h = self._sigmoid(X @ theta)
        gradient = 1/m * (X.T @ (y - h))
        return gradient
    
    def _compute_hessian(self, X, y, theta):
        m = len(y)
        h = self._sigmoid(X @ theta)

        S = h * (1 - h)
        hessian = (1/m) * ((X.T @ (S[:, np.newaxis] * X))) 
        
        # Add small regularization if Hessian is not invertible
        if np.linalg.det(hessian) < 1e-10:
            hessian += 1e-10 * np.eye(theta.shape[0])
        
        return hessian
    
    def _check_convergence(self, gradient, hessian, epsilon=1e-6):
        try:
            delta = np.linalg.solve(hessian, gradient)
            newton_decrement = np.sqrt(gradient.T @ delta)
            return newton_decrement <= epsilon
        except np.linalg.LinAlgError:
            return False
    
    def fit(self, X, y, learning_rate=0.01):
        # Add bias term
        X = np.c_[np.ones(len(X)), self._normalize_features(X)]
        
        # Initialize parameters
        n_features = X.shape[1]
        theta = np.zeros(n_features)
        max_iter = self.n_iter
        epsilon = self.eps
        
        # Store parameter history (excluding initial parameters)
        parameter_history = []
        
        for _ in range(max_iter):
            # Compute gradient and Hessian
            gradient = self._compute_gradient(X, y, theta)
            hessian = self._compute_hessian(X, y, theta)
            
            # Check convergence
            if self._check_convergence(gradient, hessian, epsilon):
                break
            
            # Update parameters using Newton's method
            try:
                delta = np.linalg.solve(hessian, gradient) # Solve the equation H * delta = g
                theta = theta + learning_rate * delta
                parameter_history.append(theta)
            except np.linalg.LinAlgError:
                break
        
        self.theta = theta
        return np.array(parameter_history)
    
    def predict(self, X):
        if self.theta is None:
            raise ValueError("Model has not been fitted yet.")
        
        # Add bias term and normalize features
        X = np.c_[np.ones(len(X)), self._normalize_features(X)] # same normalization as in fit
        
        # Compute probabilities
        probabilities = self._sigmoid(X @ self.theta)
        
        # Convert probabilities to class labels
        return (probabilities >= 0.5).astype(int)

import matplotlib.pyplot as plt
def plot_decision_boundary(model, X, y):
    plt.figure(figsize=(10, 8))
    
    # First, normalize the input features using the model's stored parameters
    X_norm = (X - model.mean) / model.std

    # Plot the normalized data points
    plt.scatter(X_norm[y == 0, 0], X_norm[y == 0, 1], c='blue', marker='o', label='Class 0')
    plt.scatter(X_norm[y == 1, 0], X_norm[y == 1, 1], c='red', marker='x', label='Class 1')

    # Create grid points in normalized space
    x_min, x_max = X_norm[:, 0].min() - 1, X_norm[:, 0].max() + 1

    # In normalized space, the decision boundary is simpler:
    # theta[0] + theta[1]*x1 + theta[2]*x2 = 0
    # Solving for x2: x2 = -(theta[0] + theta[1]*x1)/theta[2]
    x1_points = np.array([x_min, x_max])
    x2_points = -(model.theta[0] + model.theta[1]*x1_points)/model.theta[2]

    # Plot the decision boundary
    plt.plot(x1_points, x2_points, 'g-', label='Decision Boundary')

    plt.xlabel('Normalized x₁')
    plt.ylabel('Normalized x₂')
    plt.title('Logistic Regression Decision Boundary (Normalized Space)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    # Add the equation of the decision boundary to the plot for clarity
    equation = f'Decision boundary: {model.theta[2]:.2f}x₂ = -{model.theta[0]:.2f} - {model.theta[1]:.2f}x₁'
    plt.text(0.05, 0.95, equation, transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8))

    return plt