import cvxopt
import numpy as np

class SupportVectorMachine:
    '''
    Binary Classifier using Support Vector Machine
    '''
    def __init__(self):
        self.alphas = None
        self.sv_indices = None
        self.w = None 
        self.b = None
        self.svalphas = None
        self.svX = None
        self.svY = None
        self.kernel = None
        self.gamma = None
        
    def fit(self, X, y, kernel = 'linear', C = 1.0, gamma = 0.001):
        '''
        Learn the parameters from the given training data
        Classes are 0 or 1
        
        Args:
            X: np.array of shape (N, D) 
                where N is the number of samples and D is the flattened dimension of each image
                
            y: np.array of shape (N,)
                where N is the number of samples and y[i] is the class of the ith sample
                
            kernel: str
                The kernel to be used. Can be 'linear' or 'gaussian'
                
            C: float
                The regularization parameter
                
            gamma: float
                The gamma parameter for gaussian kernel, ignored for linear kernel
        '''
        self.kernel = kernel
        self.gamma = gamma  # Store gamma value for later use
        
        if kernel == 'linear':
            self.fit_linear(X, y, C)
        elif kernel == 'gaussian':
            self.fit_gaussian(X, y, C, gamma)
        else:
            raise ValueError('Invalid kernel type')
        
    def predict(self, X):
        '''
        Predict the class of the input data
        
        Args:
            X: np.array of shape (N, D) 
                where N is the number of samples and D is the flattened dimension of each image
                
        Returns:
            np.array of shape (N,)
                where N is the number of samples and y[i] is the class of the
                ith sample (0 or 1)
        '''
        if self.kernel == 'linear':
            return self.predict_linear(X)
        elif self.kernel == 'gaussian':
            first,_ = self.predict_gaussian(X)
            return first
        else:
            raise ValueError('Invalid kernel type')

    def fit_linear(self, X, y, C):
        '''
        Learn the parameters from the given training data using linear kernel
        
        Args:
            X: np.array of shape (N, D) 
                where N is the number of samples and D is the flattened dimension of each image
                
            y: np.array of shape (N,)
                where N is the number of samples and y[i] is the class of the ith sample
                
            C: float
                The regularization parameter
        '''
        y = np.where(y == 0, -1, y)

        # Number of training samples
        N = X.shape[0]

        # Compute the Gram matrix (P = Y Y^T * X X^T)
        P = cvxopt.matrix(np.outer(y, y) * np.dot(X, X.T))  # P = Y Y^T * K(X, X)
        q = cvxopt.matrix(-np.ones(N))  # q = -1

        # Inequality constraints (0 <= alpha <= C)
        G = cvxopt.matrix(np.vstack((-np.eye(N), np.eye(N))))  # G = [-I; I]
        h = cvxopt.matrix(np.hstack((np.zeros(N), np.ones(N) * C)))  # h = [0; C]

        # Equality constraint (sum(alpha_i * y_i) = 0)
        A = cvxopt.matrix(y.astype(float), (1, N))  # A = y^T
        b = cvxopt.matrix(0.0)  # b = 0

        # Solve the QP problem
        cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        self.alphas = np.ravel(solution['x'])  # Extract alpha values

        # Support vectors have non zero lagrange multipliers
        self.sv_indices = self.alphas > 1e-5

        self.svalphas = self.alphas[self.sv_indices]
        self.svX = X[self.sv_indices]
        self.svY = y[self.sv_indices]

        # Compute the weight vector w
        self.w = np.sum(self.svalphas[:, None] * self.svY[:, None] * self.svX, axis=0)

        # Compute the bias b
        self.b = np.mean(self.svY - np.dot(self.svX, self.w))

    def gaussian_kernel_matrix(self, X1, X2=None):
        """
        Compute the Gaussian kernel matrix between X1 and X2.
        
        Args:
            X1: np.array of shape (n_samples_1, n_features)
            X2: np.array of shape (n_samples_2, n_features) or None
                If None, compute the kernel matrix for X1 with itself
                
        Returns:
            K: np.array of shape (n_samples_1, n_samples_2)
                The kernel matrix
        """
        if X2 is None:
            X2 = X1
            
        # Compute squared Euclidean distances efficiently
        X1_norm = np.sum(X1**2, axis=1).reshape(-1, 1)
        X2_norm = np.sum(X2**2, axis=1).reshape(1, -1)
        K = X1_norm + X2_norm - 2 * np.dot(X1, X2.T)
        
        # Apply the Gaussian kernel
        return np.exp(-self.gamma * K)

    def fit_gaussian(self, X, y, C, gamma):
        '''
        Learn the parameters from the given training data using gaussian kernel
        
        Args:
            X: np.array of shape (N, D) 
                where N is the number of samples and D is the flattened dimension of each image
                
            y: np.array of shape (N,)
                where N is the number of samples and y[i] is the class of the ith sample
                
            C: float
                The regularization parameter
                
            gamma: float
                The gamma parameter for gaussian kernel
        '''
        y = np.where(y == 0, -1, y)

        # Number of training samples
        N = X.shape[0]

        # Compute the Gaussian kernel matrix
        K = self.gaussian_kernel_matrix(X)
        
        # Set up the QP problem
        P = cvxopt.matrix(np.outer(y, y) * K)  # P = y_i * y_j * K(x_i, x_j)
        q = cvxopt.matrix(-np.ones(N))  # q = -1
        G = cvxopt.matrix(np.vstack((-np.eye(N), np.eye(N))))  # G = [-I; I]
        h = cvxopt.matrix(np.hstack((np.zeros(N), np.ones(N) * C)))  # h = [0; C]
        A = cvxopt.matrix(y.astype(float), (1, N))  # A = y^T
        b = cvxopt.matrix(0.0)  # b = 0

        # Solve the QP problem
        cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        self.alphas = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        self.sv_indices = self.alphas > 1e-5

        self.svalphas = self.alphas[self.sv_indices]
        self.svX = X[self.sv_indices]
        self.svY = y[self.sv_indices]
        
        # Compute the bias term b using support vectors
        # Get kernel values between support vectors
        sv_kernel = self.gaussian_kernel_matrix(self.svX)
        
        # Compute predictions for support vectors
        sv_predictions = np.sum(self.svalphas * self.svY * sv_kernel, axis=0)
        
        # Bias is the average difference between label and decision function for all support vectors
        self.b = np.mean(self.svY - sv_predictions)

    def predict_linear(self, X):
        '''
        Predict the class of the input data using linear kernel
        
        Args:
            X: np.array of shape (N, D) 
                where N is the number of samples and D is the flattened dimension of each image
                
        Returns:
            np.array of shape (N,)
                where N is the number of samples and y[i] is the class of the
                ith sample (0 or 1)
        '''
        # Compute decision function: f(x) = w^T x + b
        decision_values = np.dot(X, self.w) + self.b

        # Assign class labels (-1 or 1)
        predictions = np.sign(decision_values)

        # Convert back to 0 and 1 for consistency
        return np.where(predictions == -1, 0, 1)
    
    def predict_gaussian(self, X):
        '''
        Predict the class of the input data using gaussian kernel
        
        Args:
            X: np.array of shape (N, D) 
                where N is the number of samples and D is the flattened dimension of each image
                
        Returns:
            np.array of shape (N,)
                where N is the number of samples and y[i] is the class of the
                ith sample (0 or 1)'
        '''
        # Compute the kernel matrix between test samples and support vectors
        K = self.gaussian_kernel_matrix(X, self.svX)
        
        # Compute decision values for all test samples at once
        decision_values = np.dot(K, self.svalphas * self.svY) + self.b
        
        # Assign class labels (-1 or 1)
        predictions = np.sign(decision_values)

        # Convert back to 0 and 1 for consistency
        return np.where(predictions == -1, 0, 1), decision_values
    