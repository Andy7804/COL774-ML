import numpy as np
from sampling_sgd import generate

np.random.seed(42)
N = 1000000
input_mean = np.array([3, 5])  # Mean for two features
input_sigma = np.array([1, 1])  # Standard deviation
noise_sigma = 0.5  # Noise level
theta_true = np.array([3, 1, 2])  # True theta: [bias, w1, w2]

# Generate dataset
X, y = generate(N, theta_true, input_mean, input_sigma, noise_sigma)
print(np.shape(X))
print(np.shape(y))

# Get the number of samples
num_samples = X.shape[0]

# Shuffle the indices
shuffled_indices = np.random.permutation(num_samples)

# Define the split index (85% train, 15% test)
split_idx = int(num_samples * 0.80)

# Split the indices
train_indices = shuffled_indices[:split_idx]
test_indices = shuffled_indices[split_idx:]

# Create train and test sets
X_train, y_train = X[train_indices], y[train_indices]
X_test, y_test = X[test_indices], y[test_indices]

from sampling_sgd import StochasticLinearRegressor
model = StochasticLinearRegressor()
model.fit(X_train, y_train)
thetas = model.thetas
print(f"Theta for batch size 1: {thetas[0]}")
print(f"Theta for batch size 80: {thetas[1]}")
print(f"Theta for batch size 8000: {thetas[2]}")
print(f"Theta for batch size 800000: {thetas[3]}")

import numpy as np

# Add bias term (column of ones) to X
X_bias = np.insert(X, 0, 1, axis=1)  # Inserts a column of ones at index 0

# Compute theta using the Normal Equation
theta_closed_form = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y
  
print("Theta computed using closed-form solution:")
print(theta_closed_form)

y_predict = model.predict(X_test)
MSE_vals = []
for i in range(len(y_predict)):
    mse = np.mean((y_predict[i] - y_test) ** 2)
    MSE_vals.append(mse.copy())
print(f"Test Error for batch size 1: {MSE_vals[0]}")
print(f"Test Error for batch size 80: {MSE_vals[1]}")
print(f"Test Error for batch size 8000: {MSE_vals[2]}")
print(f"Test Error for batch size 800000: {MSE_vals[3]}")

y_train_p = model.predict(X_train)
MSE_vals_t = []
for i in range(len(y_train_p)):
    mse = np.mean((y_train_p[i] - y_train) ** 2)
    MSE_vals_t.append(mse.copy())
print(f"Train Error for batch size 1: {MSE_vals_t[0]}")
print(f"Train Error for batch size 80: {MSE_vals_t[1]}")
print(f"Train Error for batch size 8000: {MSE_vals_t[2]}")
print(f"Train Error for batch size 800000: {MSE_vals_t[3]}")

time = model.times
for i in range(len(time)):
    print(f"Time taken for batch size {model.batch_sizes[i]}: {time[i]}")

x = model.thetas_hist

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
from matplotlib.lines import Line2D   # For creating custom legend elements

# Assuming model is an instance of StochasticLinearRegressor and model.thetas_hist is computed
# model.thetas_hist is a list of numpy arrays, each of shape (num_iterations, 3)

theta_history_list = model.thetas_hist

# Create a new figure for 3D animation
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("3D Trajectory of Theta Updates (Stochastic Linear Regressor)")
ax.set_xlabel("Theta 0 (Bias)")
ax.set_ylabel("Theta 1 (Weight 1)")
ax.set_zlabel("Theta 2 (Weight 2)")

# Define colors for each run
colors = ['b', 'g', 'r', 'orange']

# Create custom legend elements for the batch sizes
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Batch size 1',
           markerfacecolor='b', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Batch size 80',
           markerfacecolor='g', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Batch size 8000',
           markerfacecolor='r', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Batch size 800000',
           markerfacecolor='orange', markersize=10)
]
ax.legend(handles=legend_elements)

# Determine the maximum number of iterations over all runs
max_iters = max(traj.shape[0] for traj in theta_history_list)

# Create a text annotation to show the current iteration
text_handle = ax.text2D(0.05, 0.95, "", transform=ax.transAxes, fontsize=12,
                          bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

# We'll store the current scatter point handles so we can remove them on each iteration
scatter_handles = [None] * len(theta_history_list)

# Animation loop: Iterate over the maximum number of iterations.
for i in range(max_iters):
    # For each run (each set of theta updates corresponding to a batch size)
    for j, traj in enumerate(theta_history_list):
        # If the current iteration exceeds the run's length, use its final value
        current_pt = traj[i] if i < traj.shape[0] else traj[-1]

        # Remove previous scatter point if it exists
        if scatter_handles[j] is not None:
            try:
                scatter_handles[j].remove()
            except Exception:
                pass

        # Plot the current position as a scatter point
        scatter_handles[j] = ax.scatter(current_pt[0], current_pt[1], current_pt[2],
                                        color=colors[j % len(colors)], s=100)

        # Optionally, plot the trajectory so far for run j (as a line)
        traj_so_far = traj[:min(i + 1, traj.shape[0])]
        ax.plot(traj_so_far[:, 0], traj_so_far[:, 1], traj_so_far[:, 2],
                color=colors[j % len(colors)], alpha=0.5)

    # Update the annotation text with the current iteration number
    text_handle.set_text(f"Iteration: {i}")

    # Redraw the figure and pause briefly
    plt.draw()
    plt.pause(0.2)  # Adjust pause for animation speed

plt.show()