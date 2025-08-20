# Imports - you can add any other permitted libraries
import numpy as np
import time
# You may add any other functions to make your code more modular. However,
# do not change the function signatures (name and arguments) of the given functions,
# as these functions will be called by the autograder.

def generate(N, theta, input_mean, input_sigma, noise_sigma):
    # Standard notation: m = #samples, n = #features
    m = N
    n = len(input_mean)  # Assuming input_mean and input_sigma are consistent

    # Generate input data
    X = np.random.normal(input_mean, input_sigma, (m, n))  # Shape: (N, num_features)

    # Add bias term
    X_bias = np.insert(X, 0, 1, axis=1)  # Shape: (N, num_features + 1)

    # Ensure theta is a 1D array
    theta = np.asarray(theta).reshape(-1)  # Shape: (num_features + 1,)

    # Generate noise and ensure it is a 1D array
    noise = np.random.normal(0, noise_sigma, (m,)).reshape(-1)  # Shape: (N,)

    # Compute y
    y = np.dot(X_bias, theta) + noise  # Shape: (N,)

    return X, y  # X: (N, num_features), y: (N,)

class StochasticLinearRegressor:
    def __init__(self):
        self.batch_sizes = [1, 80, 8000, 800000] 
        self.thetas_hist = None
        self.thetas = None
        self.times = None  # Store convergence times

        self.eps = [1e-5, 1e-5, 1e-5, 1e-5]
        self.n_iter = [100000, 100000, 100000, 100000]
        self.k = [100, 1000, 1000, 10]

    def fit(self, X, y, learning_rate=0.001):
        self.thetas_hist = []
        self.thetas = []
        self.times = []  # Initialize times list

        for i, batch_size in enumerate(self.batch_sizes):
            n_iter, eps, k = self.n_iter[i], self.eps[i], self.k[i]

            # Start measuring time
            start_time = time.time()

            hist_theta, hist_loss, hist_theta_epoch, hist_loss_epoch, hist_theta_k, hist_loss_k = \
                self.fit_helper(X, y, batch_size, n_iter, eps, k, learning_rate)

            # Stop measuring time
            elapsed_time = time.time() - start_time
            self.times.append(elapsed_time)  # Store elapsed time

            self.thetas_hist.append(hist_theta_epoch.copy())
            self.thetas.append(hist_theta_epoch[-1].copy())

        return self.thetas_hist

    # helper function to perform gradient descent for a given set of hyperparameters
    def fit_helper(self, X, y, batch_size, n_iter, eps, k, learning_rate=0.001):
        m, n = X.shape
        r = batch_size

        indices = np.random.permutation(m)  # Get random indices
        X, y = X[indices], y[indices]  # Shuffle X and Y using the same indices

        # Initialize parameters to 0
        theta = np.zeros(n+1) 

        # Initialize model history - per SGD iteration
        hist_theta = [] 
        hist_loss = [] 

        # Initialize model history - per epoch
        hist_theta_epoch = []
        hist_loss_epoch = []

        # Initialize model history - per k batches (for convergence)
        hist_theta_k = []
        hist_loss_k = []
        
        # Add a column of ones to X (i.e. X0 = 1) to the left 
        X_bias = np.insert(X, 0, 1, axis=1) 

        batch_counter = 0 # number of batches processed 
        epoch_counter = 0 # number of epochs processed

        k_loss = [] # will be used to calculate average loss per k batches (for convergence purposes)
        converged = False
        while (epoch_counter < n_iter and not converged):
            epoch_loss = [] # will be used to calculate average loss per epoch (for completion purposes)
            for batch_index in range(0, m, r):
                
                # Get current batch
                X_batch = X_bias[batch_index:batch_index+r]
                y_batch = y[batch_index:batch_index+r].reshape(-1)

                # Gradient calculation and updating parameters
                y_hat = np.dot(X_batch, theta) 
                gradient = np.dot(X_batch.T, (y_hat - y_batch)) / r
                theta -= learning_rate * gradient

                # Update model history (one SGD iteration)
                current_loss = 1/(2*r) * np.sum((y_batch - y_hat)**2)
                hist_theta.append(theta.copy())
                hist_loss.append(current_loss.copy())

                # epoch and k losses are updated which will be used for averaging
                epoch_loss.append(current_loss.copy())
                k_loss.append(current_loss.copy())

                batch_counter += 1 # increment batch counter

                # Check convergence (if k batches done)
                if batch_counter > 0 and batch_counter % k == 0:
                    avg_loss = np.mean(k_loss.copy()) # average loss per k batches
                    hist_loss_k.append(avg_loss.copy())
                    hist_theta_k.append(theta.copy())

                    k_loss = []

                    if len(hist_loss_k) > 1:
                        if abs(hist_loss_k[-1] - hist_loss_k[-2]) <= eps:
                            converged = True
                            break
            
            # Update model history (one epoch)  
            avg_loss = np.mean(epoch_loss.copy())
            hist_loss_epoch.append(avg_loss.copy())
            hist_theta_epoch.append(theta.copy())

            epoch_counter += 1 # increment epoch counter
            if converged:
                break

        return np.array(hist_theta), np.array(hist_loss), np.array(hist_theta_epoch), np.array(hist_loss_epoch), np.array(hist_theta_k), np.array(hist_loss_k)

    def predict(self, X):
        predictions = []
        for i,theta in enumerate(self.thetas):
            # Use the final parameters from each batch size's training
            theta = self.thetas[i]
            predictions.append(self.predict_helper(X, theta).copy())
        
        return predictions
    
    # similar helper function
    def predict_helper(self, X, theta):
        X_bias = np.insert(X, 0, 1, axis=1)
        y_hat = np.dot(X_bias, theta)
        return y_hat