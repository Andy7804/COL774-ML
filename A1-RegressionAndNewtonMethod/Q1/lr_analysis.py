import numpy as np
import pandas as pd
from pathlib import Path
from linear_regression import LinearRegressor
import matplotlib.pyplot as plt

def get_theta():
    # (1) Import the necessary libraries
    import numpy as np
    import pandas as pd
    from pathlib import Path

    # (2) Get the dataset from the CSV files (relative to the ASSIGNMENT1 folder)
    # Get the absolute path of the ASSIGNMENT1 folder (i.e. the parent folder)
    parent_folder = Path.cwd().parent  # Moves up from Q4 to ASSIGNMENT1

    # Define the data folder relative to ASSIGNMENT1
    data_folder = parent_folder / "data" / "Q1"

    # Define file paths for q4x.dat and q4y.dat (case-sensitive filenames)
    x_file = data_folder / "linearX.csv"
    y_file = data_folder / "linearY.csv"

    # Load the datasets - CSV to Pandas df to Numpy array
    X = pd.read_csv(x_file, header=None).values  
    Y = pd.read_csv(y_file, header=None).values.ravel()

    from linear_regression import LinearRegressor

    model = LinearRegressor()
    model.fit(X, Y)
    theta_0, theta_1 = model.parameters  # Extract parameters
    print(f"Equation of the regression line: y = {theta_0:.2f} + {theta_1:.2f}x")

def plot_hypothesis():
    parent_folder = Path.cwd().parent  # Moves up from Q4 to ASSIGNMENT1

    # Define the data folder relative to ASSIGNMENT1
    data_folder = parent_folder / "data" / "Q1"

    # Define file paths for q4x.dat and q4y.dat (case-sensitive filenames)
    x_file = data_folder / "linearX.csv"
    y_file = data_folder / "linearY.csv"

    # Load the datasets - CSV to Pandas df to Numpy array
    X = pd.read_csv(x_file, header=None).values  
    Y = pd.read_csv(y_file, header=None).values.ravel()

    model = LinearRegressor()
    model.fit(X, Y)
    theta_0, theta_1 = model.parameters  # Extract parameters
    print(f"Equation of the regression line: y = {theta_0:.2f} + {theta_1:.2f}x")

    # (2) Plot the data and the hypothesis function
    plt.figure(figsize=(8,6))
    plt.scatter(X, Y, color='blue', label='Data')
    plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression Line')
    plt.xlabel('Acidity')
    plt.ylabel('Density of wine')
    plt.title('Linear Regression Fit')
    plt.legend()
    plt.show()

def plot_gradient_descent():
    # Get the absolute path of the ASSIGNMENT1 folder (i.e. the parent folder)
    parent_folder = Path.cwd().parent  # Moves up from Q4 to ASSIGNMENT1

    # Define the data folder relative to ASSIGNMENT1
    data_folder = parent_folder / "data" / "Q1"

    # Define file paths for q4x.dat and q4y.dat (case-sensitive filenames)
    x_file = data_folder / "linearX.csv"
    y_file = data_folder / "linearY.csv"

    # Load the datasets - CSV to Pandas df to Numpy array
    X = pd.read_csv(x_file, header=None).values  
    Y = pd.read_csv(y_file, header=None).values.ravel()

    from linear_regression import LinearRegressor

    model = LinearRegressor()
    model.fit(X, Y)
    theta_0, theta_1 = model.parameters  # Extract parameters
    print(f"Equation of the regression line: y = {theta_0:.2f} + {theta_1:.2f}x")

    x = model.loss_history
    y = model.parameters_history

    import time
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

    # -------------------------------------------------------------------------
    # NOTE: Ensure that the following variables are defined appropriately:
    #       - model : your regression model instance (with attributes like
    #                 parameters_history and a method MSEloss)
    #       - X, Y  : your data arrays
    #
    # For example, you might load your model and data before this snippet.
    # -------------------------------------------------------------------------

    # Convert parameter history to a NumPy array
    theta_history = np.array(model.parameters_history)  # shape: (iterations, 2)
    theta = model.parameters  # final parameters

    # Create a meshgrid for theta0 and theta1 values around the final parameters
    theta0_range = np.linspace(theta[0] - 25, theta[0] + 25, 100)
    theta1_range = np.linspace(theta[1] - 25, theta[1] + 25, 100)
    theta0_mesh, theta1_mesh = np.meshgrid(theta0_range, theta1_range)

    # Prepare the design matrix by adding the bias term (column of ones)
    X_bias = np.insert(X, 0, 1, axis=1)

    # Compute loss values over the meshgrid using the model's MSEloss
    loss_values = np.array([
        [model.MSEloss(X_bias, Y, np.array([t0, t1])) for t1 in theta1_range]
        for t0 in theta0_range
    ])

    # Create one figure and axes for the animation
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle("Gradient Descent", fontsize=16)  # Display title on top of the figure
    ax = fig.add_subplot(111, projection='3d')

    # Plot the static 3D loss surface only once
    surface = ax.plot_surface(theta0_mesh, theta1_mesh, loss_values, cmap='viridis', alpha=0.7)
    fig.colorbar(surface, shrink=0.75, pad=0.1)

    ax.set_xlabel('Theta 0')
    ax.set_ylabel('Theta 1')
    ax.set_zlabel('Loss Function (J)')
    ax.view_init(elev=30, azim=135)

    # Create a text annotation object to display iteration info
    text_handle = ax.text2D(
        0.05, 0.95, "", transform=ax.transAxes,
        fontsize=12, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    # Initialize a handle for the blue (current) point
    scatter_blue = None

    # Animation Loop: update the existing figure without closing it.
    for i in range(len(theta_history)):
        # Convert previous blue point to red by plotting it (if not the first iteration)
        if i > 0:
            pt_prev = theta_history[i - 1]
            z_prev = model.MSEloss(X_bias, Y, pt_prev)
            ax.scatter(pt_prev[0], pt_prev[1], z_prev, color='r', s=50)

        # Remove the previous blue point if it exists
        if scatter_blue is not None:
            scatter_blue.remove()

        # Plot the current point in blue
        current_pt = theta_history[i]
        current_loss = model.MSEloss(X_bias, Y, current_pt)
        scatter_blue = ax.scatter(current_pt[0], current_pt[1], current_loss, color='b', s=100)

        # Update the text annotation with current values
        info_text = (f"Iteration: {i}\n"
                    f"Theta0: {current_pt[0]:.2f}\n"
                    f"Theta1: {current_pt[1]:.2f}\n"
                    f"Loss: {current_loss:.2f}")
        text_handle.set_text(info_text)

        # Redraw the canvas and pause briefly
        plt.draw()
        plt.pause(0.2)

    plt.show()

def plot_contour():
    # (1) Import the necessary libraries
    import numpy as np
    import pandas as pd
    from pathlib import Path

    # (2) Get the dataset from the CSV files (relative to the ASSIGNMENT1 folder)
    # Get the absolute path of the ASSIGNMENT1 folder (i.e. the parent folder)
    parent_folder = Path.cwd().parent  # Moves up from Q4 to ASSIGNMENT1

    # Define the data folder relative to ASSIGNMENT1
    data_folder = parent_folder / "data" / "Q1"

    # Define file paths for q4x.dat and q4y.dat (case-sensitive filenames)
    x_file = data_folder / "linearX.csv"
    y_file = data_folder / "linearY.csv"

    # Load the datasets - CSV to Pandas df to Numpy array
    X = pd.read_csv(x_file, header=None).values  
    Y = pd.read_csv(y_file, header=None).values.ravel()

    model = LinearRegressor()
    model.fit(X, Y)
    theta_0, theta_1 = model.parameters  # Extract parameters
    print(f"Equation of the regression line: y = {theta_0:.2f} + {theta_1:.2f}x")

    x = model.loss_history
    y = model.parameters_history

    import time
    import numpy as np
    import matplotlib.pyplot as plt

    # -------------------------------------------------------------------------
    # NOTE: Ensure that the following variables are defined appropriately:
    #       - model : your regression model instance (with attributes like
    #                 parameters_history and a method MSEloss)
    #       - X, Y  : your data arrays
    #
    # For example, you might load your model and data before this snippet.
    # -------------------------------------------------------------------------

    # Convert parameter history to a NumPy array
    theta_history = np.array(model.parameters_history)  # shape: (iterations, 2)
    theta = model.parameters  # final parameters

    # Create a meshgrid for theta0 and theta1 values around the final parameters
    theta0_range = np.linspace(theta[0] - 25, theta[0] + 25, 100)
    theta1_range = np.linspace(theta[1] - 25, theta[1] + 25, 100)
    theta0_mesh, theta1_mesh = np.meshgrid(theta0_range, theta1_range)

    # Prepare the design matrix by adding the bias term (column of ones)
    X_bias = np.insert(X, 0, 1, axis=1)

    # Compute loss values over the meshgrid using the model's MSEloss
    loss_values = np.array([
        [model.MSEloss(X_bias, Y, np.array([t0, t1])) for t1 in theta1_range]
        for t0 in theta0_range
    ])

    # Create one figure and axes for the animation (2D contour mode)
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle("Gradient Descenet", fontsize=16)  # Persistent title at top

    # Animation Loop: update the same figure (clear axes each iteration)
    for i in range(len(theta_history)):
        ax.cla()  # Clear the previous plot

        # Reapply labels and title after clearing axes
        ax.set_xlabel("Theta 0")
        ax.set_ylabel("Theta 1")
        ax.set_title("Gradient Descenet")
        
        # Plot the full contour plot of the loss function
        contour_set = ax.contour(theta0_mesh, theta1_mesh, loss_values, levels=50, cmap='viridis')
        # Optionally, if you prefer a filled contour, you can use:
        # contour_set = ax.contourf(theta0_mesh, theta1_mesh, loss_values, levels=50, cmap='viridis')
        
        # Plot previous iterations' points in red (if any)
        if i > 0:
            ax.scatter(theta_history[:i, 0], theta_history[:i, 1], color='r', s=50, label='Previous')
        
        # Plot current iteration point in blue
        current_pt = theta_history[i]
        current_loss = model.MSEloss(X_bias, Y, current_pt)
        ax.scatter(current_pt[0], current_pt[1], color='b', s=100, label='Current')
        
        # Highlight the contour corresponding to the current loss value with a dashed white line
        ax.contour(theta0_mesh, theta1_mesh, loss_values, levels=[current_loss], colors='w', linestyles='--')
        
        # Set axis limits for consistency
        ax.set_xlim(theta0_range[0], theta0_range[-1])
        ax.set_ylim(theta1_range[0], theta1_range[-1])
        
        # Add text annotation with current iteration info
        info_text = (f"Iteration: {i}\n"
                    f"Theta0: {current_pt[0]:.2f}\n"
                    f"Theta1: {current_pt[1]:.2f}\n"
                    f"Loss: {current_loss:.2f}")
        ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Redraw the canvas and pause before next iteration
        plt.draw()
        plt.pause(0.2)

    plt.show()

def plot_contour_3():
    import numpy as np
    import pandas as pd
    from pathlib import Path
    import matplotlib.pyplot as plt
    from linear_regression import LinearRegressor

    # (2) Get the dataset from the CSV files (relative to the ASSIGNMENT1 folder)
    # Get the absolute path of the ASSIGNMENT1 folder (i.e. the parent folder)
    parent_folder = Path.cwd().parent  # Moves up from Q4 to ASSIGNMENT1

    # Define the data folder relative to ASSIGNMENT1
    data_folder = parent_folder / "data" / "Q1"

    # Define file paths for q4x.dat and q4y.dat (case-sensitive filenames)
    x_file = data_folder / "linearX.csv"
    y_file = data_folder / "linearY.csv"

    # Load the datasets - CSV to Pandas df to Numpy array
    X = pd.read_csv(x_file, header=None).values  
    Y = pd.read_csv(y_file, header=None).values.ravel()

    # -----------------------------
    # (2) Run gradient descent for three learning rates
    # -----------------------------
    learning_rates = [0.001, 0.025, 0.1]
    trajectories = {}   # Dictionary to store parameter histories keyed by the learning rate
    final_params_list = []  # To compute a common meshgrid range

    for lr in learning_rates:
        model = LinearRegressor()
        # Call fit with the current learning rate. (Other hyperparameters remain default.)
        model.fit(X, Y, learning_rate=lr)
        # Store the trajectory as a NumPy array (each row is [theta0, theta1]).
        traj = np.array(model.parameters_history)
        trajectories[lr] = traj
        final_params_list.append(traj[-1])
        print(f"Learning rate {lr} final parameters: {traj[-1]}")

    # Compute the center point (average of the three final parameter values)
    final_params_array = np.array(final_params_list)
    center = final_params_array.mean(axis=0)

    # -----------------------------
    # (3) Create a meshgrid based on the center of all trajectories
    # -----------------------------
    # You can adjust the margin if needed.
    margin = 25
    theta0_range = np.linspace(center[0] - margin, center[0] + margin, 100)
    theta1_range = np.linspace(center[1] - margin, center[1] + margin, 100)
    theta0_mesh, theta1_mesh = np.meshgrid(theta0_range, theta1_range)

    # Prepare the design matrix by adding the bias term (a column of ones)
    X_bias = np.insert(X, 0, 1, axis=1)

    # Define a loss function (mean squared error) that matches your model's MSEloss.
    def compute_loss(params):
        predictions = X_bias.dot(params)
        return np.mean((predictions - Y)**2)

    # Compute loss values over the meshgrid
    loss_values = np.array([
        [compute_loss(np.array([t0, t1])) for t1 in theta1_range]
        for t0 in theta0_range
    ])

    # -----------------------------
    # (4) Animate the Trajectories on the Contour Plot
    # -----------------------------
    # Determine the maximum number of iterations over the three runs.
    max_iter = max(traj.shape[0] for traj in trajectories.values())

    # Define colors for each learning rate for consistency.
    colors = {0.001: 'b', 0.025: 'g', 0.1: 'r'}

    # Create one figure and single set of axes.
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle("Gradient Descenet", fontsize=16)  # Persistent title at top

    # Animation loop: iterate from 0 to max_iter - 1
    for i in range(max_iter):
        ax.cla()  # Clear the previous frame

        # Reapply axis labels and plot title for each frame
        ax.set_xlabel("Theta 0")
        ax.set_ylabel("Theta 1")
        ax.set_title("Gradient Descenet")
        
        # Plot the complete loss function contour in the background
        ax.contour(theta0_mesh, theta1_mesh, loss_values, levels=50, cmap='viridis')
        
        # Initialize text for current iteration info
        info_text = f"Iteration: {i}\n"
        
        # For each learning rate, plot the trajectories up to the current iteration
        for lr in learning_rates:
            traj = trajectories[lr]
            # Determine the number of points to show for this trajectory:
            current_idx = min(i, traj.shape[0] - 1)
            # Plot the trajectory line (up to the current point) in the defined color.
            ax.plot(traj[:current_idx + 1, 0], traj[:current_idx + 1, 1],
                    color=colors[lr], marker='o', label=f"lr = {lr}")
            # Plot all previous points as red (if any) and current point as blue.
            if current_idx > 0:
                ax.scatter(traj[:current_idx, 0], traj[:current_idx, 1],
                        color='r', s=30)
            current_pt = traj[current_idx]
            # Compute the loss value at the current point
            current_loss = compute_loss(current_pt)
            # Draw current point in blue (to highlight it)
            ax.scatter(current_pt[0], current_pt[1], color='b', s=100)
            # Highlight the contour corresponding to the current loss value with a dashed white line.
            ax.contour(theta0_mesh, theta1_mesh, loss_values, levels=[current_loss],
                    colors='w', linestyles='--')
            # Append current details to the annotation text
            info_text += (f"lr={lr}: theta0={current_pt[0]:.2f}, "
                        f"theta1={current_pt[1]:.2f}, loss={current_loss:.2f}\n")
        
        # Set axis limits for consistency.
        ax.set_xlim(theta0_range[0], theta0_range[-1])
        ax.set_ylim(theta1_range[0], theta1_range[-1])
        
        # Display the current iteration information as a text box.
        ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Add the legend (only once per frame is sufficient)
        ax.legend(loc='upper right')
        
        # Update the plot and pause briefly (0.2 seconds)
        plt.draw()
        plt.pause(0.2)

    plt.show()

def main(option=None):
    """
    Execute specific functions based on the input argument.
    
    Args:
        option (int): Number between 1-5 to select which function to run
            1: get_theta()
            2: plot_hypothesis()
            3: plot_gradient_descent()
            4: plot_contour()
            5: plot_contour_3()
            None: runs all functions (default behavior)
    """
    if option is None:
        # Original behavior - run all functions
        get_theta()
        plot_hypothesis()
        plot_gradient_descent()
        plot_contour()
        plot_contour_3()
    else:
        # Convert option to int in case it's passed as string
        try:
            option = int(option)
        except ValueError:
            raise ValueError("Option must be a number between 1 and 5")
            
        # Dictionary mapping options to functions
        function_map = {
            1: get_theta,
            2: plot_hypothesis,
            3: plot_gradient_descent,
            4: plot_contour,
            5: plot_contour_3
        }
        
        # Check if option is valid
        if option not in function_map:
            raise ValueError("Option must be a number between 1 and 5")
            
        # Execute the selected function
        function_map[option]()

# Example usage:
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # If argument is provided via command line
        main(sys.argv[1])
    else:
        # No argument provided, run all functions
        main()