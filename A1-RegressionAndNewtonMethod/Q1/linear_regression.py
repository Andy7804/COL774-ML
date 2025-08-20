# Imports - you can add any other permitted libraries
import numpy as np

# You may add any other functions to make your code more modular. However,
# do not change the function signatures (name and arguments) of the given functions,
# as these functions will be called by the autograder.

class LinearRegressor:
    def __init__(self):
        # parameters - saves the state of the model
        self.parameters = None

        # hyperparameters eps and n_iter : for convergence (eps/n_iter) 
        self.eps = 1e-6
        self.n_iter = 100000

        # history of model during training
        self.parameters_history = None
        self.loss_history = None


    def MSEloss(self, X_b, y, parameters):
        m = X_b.shape[0]
        loss = 0.0
        
        y_hat = np.dot(X_b, parameters)
        y_true = y
        loss = 1/(2*m) * np.sum((y_true - y_hat)**2)
        return loss


    def eps_convergence(self):
        # Ensure there are at least two loss values to compare
        if len(self.loss_history) < 2:
            return False
        
        # Compute the absolute difference in loss values
        loss_difference = abs(self.loss_history[-1] - self.loss_history[-2])
        threshold = self.eps
        
        # Check if the loss difference is below the threshold eps (absolute comparision for now)
        return loss_difference <= threshold
    
    def fit(self, X, y, learning_rate=0.01):
        m, n = X.shape

        # Initialize parameters to 0
        parameters = np.zeros(n+1) 

        # Initialize model history
        self.parameters_history = []
        self.loss_history = []

        # Add a column of ones to X (i.e. X0 = 1) to the left 
        X_bias = np.insert(X, 0, 1, axis=1) 

        # Batch Gradient Descent
        num_iter = 0  # local variable maintaining the number of iterations
        self.parameters_history.append(parameters.copy()) # parameters_history[0] = initial parameters
        self.loss_history.append(self.MSEloss(X_bias, y, parameters.copy())) # loss_history[0] = initial loss
        while(num_iter < self.n_iter):
            # Gradient calculation and updating parameters
            y_hat = np.dot(X_bias, parameters)
            gradient = np.dot(X_bias.T, (y_hat - y)) / m
            parameters -= learning_rate * gradient

            self.parameters_history.append(parameters.copy())
            self.loss_history.append(self.MSEloss(X_bias, y, parameters.copy()))
            num_iter += 1

            # Check for convergence
            if self.eps_convergence():
                break   
        
        self.parameters = parameters.copy()
        return np.array(self.parameters_history)[1:]  # Remove the first row i.e. initial parameters

    
    def predict(self, X):
        parameters = self.parameters
        X_bias = np.insert(X, 0, 1, axis=1)
        y_hat = np.dot(X_bias, parameters)
        return y_hat