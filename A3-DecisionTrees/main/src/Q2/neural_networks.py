import os
import sys
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

def getTrainingData(train_path):
    """
    Load and preprocess training images and labels from the train_path directory.
    """
    image_list = []
    label_list = []

    for folder_name in sorted(os.listdir(train_path)):
        folder_path = os.path.join(train_path, folder_name)
        if not os.path.isdir(folder_path):
            continue
        label = int(folder_name.lstrip('0') or '0')
        for filename in os.listdir(folder_path):
            if filename.endswith((".ppm", ".png", ".jpg", ".jpeg")):
                file_path = os.path.join(folder_path, filename)
                try:
                    img = Image.open(file_path).resize((28, 28))
                    img_array = np.asarray(img).astype(np.float32) / 255.0
                    img_flat = img_array.flatten()
                    image_list.append(img_flat)
                    one_hot = np.zeros(43)
                    one_hot[label] = 1
                    label_list.append(one_hot)
                except Exception as e:
                    print(f"Error loading image {file_path}: {e}")

    X = np.array(image_list)
    y = np.array(label_list)
    return X, y

def getTestData(test_path):
    """
    Load and preprocess test images from the test_path directory.
    """
    image_list = []
    
    for filename in sorted(os.listdir(test_path)):
        if filename.endswith((".ppm", ".png", ".jpg", ".jpeg")):
            file_path = os.path.join(test_path, filename)
            try:
                img = Image.open(file_path).resize((28, 28))
                img_array = np.asarray(img).astype(np.float32) / 255.0
                img_flat = img_array.flatten()
                image_list.append(img_flat)
            except Exception as e:
                print(f"Error loading test image {file_path}: {e}")
    
    X_test = np.array(image_list)
    return X_test

def getTestOutput(annotation_file):
    """
    Load true labels from the test_labels.csv file.
    """
    label_list = []
    
    annotations = pd.read_csv(annotation_file)
    
    for _, row in annotations.iterrows():
        label = row['label']
        one_hot = np.zeros(43)
        one_hot[label] = 1
        label_list.append(one_hot)
    
    y_test = np.array(label_list)
    return y_test

class NeuralNetwork:
    def __init__(self, batchSize, numFeatures, hiddenLayerArchitecture, numTargetClasses, learningRate):
        self.batchSize = batchSize
        self.numFeatures = numFeatures
        self.hiddenLayerArchitecture = hiddenLayerArchitecture
        self.numTargetClasses = numTargetClasses
        self.learningRate = learningRate

        layerSizes = [numFeatures] + hiddenLayerArchitecture + [numTargetClasses]   
        self.weights = []
        self.biases = []

        for i in range(len(layerSizes) - 1):
            weight = np.random.randn(layerSizes[i], layerSizes[i + 1]) * np.sqrt(2.0 / layerSizes[i])
            bias = np.zeros((1, layerSizes[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)

        self.activations = []
        self.z_values = []

    def sigmoidActivation(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def sigmoidDerivative(self, x):
        sigVal = self.sigmoidActivation(x)
        return sigVal * (1 - sigVal)
    
    def softmaxActivation(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forwardPropagation(self, X):
        self.activations = [X]
        self.z_values = []
        
        for i in range(len(self.weights) - 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            activation = self.sigmoidActivation(z)
            self.activations.append(activation)
        
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        output = self.softmaxActivation(z)
        self.activations.append(output)
        return output    
    
    def computeCrossEntropyLoss(self, yTrue, yHat):
        yHat = np.clip(yHat, 1e-15, 1 - 1e-15)
        loss = -np.sum(yTrue * np.log(yHat)) / yTrue.shape[0]
        return loss
    
    def backwardPropagation(self, X, yTrue, yHat):
        batchSize = X.shape[0]
        weightGradients = [np.zeros_like(w) for w in self.weights]
        biasGradients = [np.zeros_like(b) for b in self.biases]

        delta = yHat - yTrue
        for i in range(len(self.weights) - 1, -1, -1):
            weightGradients[i] = np.dot(self.activations[i].T, delta) / batchSize
            biasGradients[i] = np.sum(delta, axis=0, keepdims=True) / batchSize
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.sigmoidDerivative(self.z_values[i - 1])
        return weightGradients, biasGradients
    
    def updateParameters(self, weightGradients, biasGradients, currentLearningRate=None):
        if(currentLearningRate is None):
            for i in range(len(self.weights)):
                self.weights[i] -= self.learningRate * weightGradients[i]
                self.biases[i] -= self.learningRate * biasGradients[i]
        else:
            for i in range(len(self.weights)):
                self.weights[i] -= currentLearningRate * weightGradients[i]
                self.biases[i] -= currentLearningRate * biasGradients[i]

    def save_best_weights(self):
        self.best_weights = [w.copy() for w in self.weights]
        self.best_biases = [b.copy() for b in self.biases]

    def load_best_weights(self):
        self.weights = [w.copy() for w in self.best_weights]
        self.biases = [b.copy() for b in self.best_biases]

    def train(self, X_train, y_train, X_val, y_val, max_epochs=200, patience=5):
        M = X_train.shape[0]
        best_val_loss = float('inf')
        epochs_no_improve = 0
        self.save_best_weights()

        for epoch in range(max_epochs):
            # shuffling on every epoch for now - dunno if it is really required or not !
            # implemented early stopping, but ig the num_epochs are so low, that it doesn't really help
            indices = np.random.permutation(M)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            for i in range(0, M, self.batchSize):
                X_batch = X_train_shuffled[i:i + self.batchSize]
                y_batch = y_train_shuffled[i:i + self.batchSize]
                yHat = self.forwardPropagation(X_batch)
                loss = self.computeCrossEntropyLoss(y_batch, yHat)
                weightGradients, biasGradients = self.backwardPropagation(X_batch, y_batch, yHat)
                self.updateParameters(weightGradients, biasGradients)

            y_val_pred = self.forwardPropagation(X_val)
            val_loss = self.computeCrossEntropyLoss(y_val, y_val_pred)
            
            print(f"Epoch {epoch+1}/{max_epochs}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_best_weights()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                self.load_best_weights()
                break

    def train_with_adaptive_lr(self, X_train, y_train, X_val, y_val, max_epochs=200, patience=5):
        M = X_train.shape[0]
        best_val_loss = float('inf')
        epochs_no_improve = 0
        self.save_best_weights()
        initial_lr = self.learningRate  # η0

        for epoch in range(max_epochs):
            # Calculate adaptive learning rate
            # Adding 1 to epoch to avoid division by zero in first epoch
            current_lr = initial_lr / np.sqrt(epoch + 1)
            
            indices = np.random.permutation(M)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            for i in range(0, M, self.batchSize):
                X_batch = X_train_shuffled[i:i + self.batchSize]
                y_batch = y_train_shuffled[i:i + self.batchSize]
                yHat = self.forwardPropagation(X_batch)
                loss = self.computeCrossEntropyLoss(y_batch, yHat)
                weightGradients, biasGradients = self.backwardPropagation(X_batch, y_batch, yHat)
                self.updateParameters(weightGradients, biasGradients, current_lr)

            y_val_pred = self.forwardPropagation(X_val)
            val_loss = self.computeCrossEntropyLoss(y_val, y_val_pred)
            
            print(f"Epoch {epoch+1}/{max_epochs}, LR: {current_lr:.6f}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_best_weights()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                self.load_best_weights()
                break

    def predict(self, X):
        return self.forwardPropagation(X)

def evaluate_model(nn, X, y, dataset_name):
    y_pred = nn.predict(X)
    y_true_classes = np.argmax(y, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    accuracy = np.mean(y_true_classes == y_pred_classes) * 100
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_classes, y_pred_classes, average=None, labels=np.arange(43), zero_division=0
    )
    
    # Calculate average metrics
    avg_precision = np.mean(precision)
    avg_recall = np.mean(recall)
    avg_f1 = np.mean(f1)
    
    print(f"\n{dataset_name} Metrics:")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")
    print("\nPer-Class Metrics:")
    print("Class | Precision | Recall | F1 Score")
    for i in range(43):
        print(f"{i:2d}    | {precision[i]:.4f}    | {recall[i]:.4f} | {f1[i]:.4f}")
    
    return precision, recall, f1, avg_f1, accuracy, y_pred_classes

def main():
    # Paths (adjust as needed)
    if len(sys.argv) != 5:
        print("Usage: python neural_network.py <train_data_path> <test_data_path> <output_folder_path> <question_part>")
        sys.exit(1)
    
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    output_folder = sys.argv[3]
    question_part = sys.argv[4]
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(42)

    annotation_file = "../../data/Q2/test_labels.csv"
    plots_output_folder = '../../output/Q2_plots'

    # Load training data
    print("Loading training data...")
    X, y = getTrainingData(train_path)
    print(f"Training data shape: {X.shape}, Labels shape: {y.shape}")

    # Manual shuffle and 80-20 split
    print("Shuffling and splitting training data...")
    N = X.shape[0]
    indices = np.random.permutation(N)
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    split_idx = int(0.8 * N)
    X_train = X_shuffled[:split_idx]
    y_train = y_shuffled[:split_idx]
    X_val = X_shuffled[split_idx:]
    y_val = y_shuffled[split_idx:]
    print(f"Train split: {X_train.shape}, Val split: {X_val.shape}")

    # Load test data
    print("Loading test data...")
    X_test = getTestData(test_path)
    y_test = getTestOutput(annotation_file)
    print(f"Test data shape: {X_test.shape}, Labels shape: {y_test.shape}")

    if(question_part == 'b'):
        # Parameters
        batch_size = 32
        num_features = 2352
        num_classes = 43
        learning_rate = 0.01
        hidden_units = [1, 5, 10, 50, 100]
        avg_f1_scores_train = []
        avg_f1_scores_test = []
        accuracy_train_list = []
        accuracy_test_list = []

        # Store predictions for 100 hidden units
        predictions_100 = None

        for units in hidden_units:
            print(f"\nExperiment with {units} hidden units")
            nn = NeuralNetwork(
                batchSize=batch_size,
                numFeatures=num_features,
                hiddenLayerArchitecture=[units],
                numTargetClasses=num_classes,
                learningRate=learning_rate
            )
            
            nn.train(X_train, y_train, X_val, y_val, max_epochs=100, patience=10)
            
            _, _, _, avg_f1_train, accuracy_train, _ = evaluate_model(nn, X_train, y_train, "Training")
            _, _, _, avg_f1_test, accuracy_test, y_pred_classes = evaluate_model(nn, X_test, y_test, "Test")
            
            avg_f1_scores_train.append(avg_f1_train)
            avg_f1_scores_test.append(avg_f1_test)
            accuracy_train_list.append(accuracy_train)
            accuracy_test_list.append(accuracy_test)

            # Save predictions for 100 hidden units
            if units == 100:
                predictions_100 = y_pred_classes

        # Save predictions to CSV
        if predictions_100 is not None:
            predictions_df = pd.DataFrame(predictions_100, columns=["predictions"])
            predictions_df.to_csv(f"predictions_{question_part}.csv", index=False)
            print("\nSaved predictions for 100 hidden units to 'predictions_b.csv'")

        # Plot F1 scores
        plt.figure(figsize=(8, 6))
        plt.plot(hidden_units, avg_f1_scores_train, marker='o', label='Training F1')
        plt.plot(hidden_units, avg_f1_scores_test, marker='s', label='Test F1')
        plt.xlabel('Number of Hidden Units')
        plt.ylabel('Average F1 Score')
        plt.title('Average F1 Score vs Hidden Units')
        plt.legend()
        plt.grid(True)
        os.makedirs(plots_output_folder, exist_ok=True)
        plt.savefig(os.path.join(plots_output_folder, f'f1_vs_hidden_units_{question_part}.png'))
        plt.close()
        # plt.show()

        # Summary
        print("\nSummary of Results:")
        print("Hidden Units | Train F1 | Test F1 | Train Accuracy | Test Accuracy")
        for units, f1_train, f1_test, acc_train, acc_test in zip(hidden_units, avg_f1_scores_train, avg_f1_scores_test, accuracy_train_list, accuracy_test_list):
            print(f"{units:11d} | {f1_train:.4f} | {f1_test:.4f} | {acc_train:.2f}% | {acc_test:.2f}%")

    elif (question_part == 'c'):
        # Parameters
        batch_size = 32
        num_features = 2352
        num_classes = 43
        learning_rate = 0.01
        hidden_layer_configs = [[512], [512, 256], [512, 256, 128], [512, 256, 128, 64]]
        depths = [len(config) for config in hidden_layer_configs]  # [1, 2, 3, 4]
        avg_f1_scores_train = []
        avg_f1_scores_test = []
        accuracy_train_list = []
        accuracy_test_list = []

        # Store predictions for [512, 256, 128, 64]
        predictions_deepest = None

        for config in hidden_layer_configs:
            depth = len(config)
            print(f"\nExperiment with hidden layers: {config} (depth={depth})")
            nn = NeuralNetwork(
                batchSize=batch_size,
                numFeatures=num_features,
                hiddenLayerArchitecture=config,
                numTargetClasses=num_classes,
                learningRate=learning_rate
            )
            
            nn.train(X_train, y_train, X_val, y_val, max_epochs=150, patience=10)
            
            _, _, _, avg_f1_train, accuracy_train, _ = evaluate_model(nn, X_train, y_train, "Training")
            _, _, _, avg_f1_test, accuracy_test, y_pred_classes = evaluate_model(nn, X_test, y_test, "Test")
            
            avg_f1_scores_train.append(avg_f1_train)
            avg_f1_scores_test.append(avg_f1_test)
            accuracy_train_list.append(accuracy_train)
            accuracy_test_list.append(accuracy_test)

            # Save predictions for [512, 256, 128, 64]
            if config == [512, 256, 128, 64]:
                predictions_deepest = y_pred_classes

        # Save predictions to CSV
        if predictions_deepest is not None:
            predictions_df = pd.DataFrame(predictions_deepest, columns=["predictions"])
            predictions_df.to_csv(f"predictions_{question_part}.csv", index=False)
            print("\nSaved predictions for hidden layers [512, 256, 128, 64] to 'predictions_c.csv'")

        # Plot F1 scores vs network depth
        plt.figure(figsize=(8, 6))
        plt.plot(depths, avg_f1_scores_train, marker='o', label='Training F1')
        plt.plot(depths, avg_f1_scores_test, marker='s', label='Test F1')
        plt.xlabel('Network Depth (Number of Hidden Layers)')
        plt.ylabel('Average F1 Score')
        plt.title('Average F1 Score vs Network Depth')
        plt.xticks(depths)
        plt.legend()
        plt.grid(True)
        os.makedirs(plots_output_folder, exist_ok=True)
        plt.savefig(os.path.join(plots_output_folder, f'f1_vs_network_depth_{question_part}.png'))
        plt.close()
        # plt.show()

        # Summary
        print("\nSummary of Results:")
        print("Hidden Layers | Depth | Train F1 | Test F1 | Train Accuracy | Test Accuracy")
        for config, depth, f1_train, f1_test, acc_train, acc_test in zip(hidden_layer_configs, depths, avg_f1_scores_train, avg_f1_scores_test, accuracy_train_list, accuracy_test_list):
            print(f"{str(config):<13} | {depth:<5} | {f1_train:.4f} | {f1_test:.4f} | {acc_train:.2f}% | {acc_test:.2f}%")

    elif (question_part == 'd'):
        # Parameters
        batch_size = 32
        num_features = 2352
        num_classes = 43
        learning_rate = 0.01  # Initial learning rate (η0)
        hidden_layer_configs = [[512], [512, 256], [512, 256, 128], [512, 256, 128, 64]]
        depths = [len(config) for config in hidden_layer_configs]  # [1, 2, 3, 4]
        avg_f1_scores_train = []
        avg_f1_scores_test = []
        accuracy_train_list = []
        accuracy_test_list = []
        training_epochs = []

        # Store predictions for [512, 256, 128, 64]
        predictions_deepest = None

        for config in hidden_layer_configs:
            depth = len(config)
            print(f"\nExperiment with hidden layers: {config} (depth={depth}) using adaptive learning rate")
            nn = NeuralNetwork(
                batchSize=batch_size,
                numFeatures=num_features,
                hiddenLayerArchitecture=config,
                numTargetClasses=num_classes,
                learningRate=learning_rate
            )
            
            # Train with adaptive learning rate
            nn.train_with_adaptive_lr(X_train, y_train, X_val, y_val, max_epochs=200, patience=5)
            
            # Track number of training epochs
            if hasattr(nn, 'epochs_trained'):
                training_epochs.append(nn.epochs_trained)
            
            _, _, _, avg_f1_train, accuracy_train, _ = evaluate_model(nn, X_train, y_train, "Training")
            _, _, _, avg_f1_test, accuracy_test, y_pred_classes = evaluate_model(nn, X_test, y_test, "Test")
            
            avg_f1_scores_train.append(avg_f1_train)
            avg_f1_scores_test.append(avg_f1_test)
            accuracy_train_list.append(accuracy_train)
            accuracy_test_list.append(accuracy_test)

            # Save predictions for [512, 256, 128, 64]
            if config == [512, 256, 128, 64]:
                predictions_deepest = y_pred_classes

        # Save predictions to CSV
        if predictions_deepest is not None:
            predictions_df = pd.DataFrame(predictions_deepest, columns=["predictions"])
            predictions_df.to_csv(f"predictions_{question_part}.csv", index=False)
            print("\nSaved predictions for hidden layers [512, 256, 128, 64] to 'predictions_d.csv'")

        # Plot F1 scores vs network depth
        plt.figure(figsize=(8, 6))
        plt.plot(depths, avg_f1_scores_train, marker='o', label='Training F1')
        plt.plot(depths, avg_f1_scores_test, marker='s', label='Test F1')
        plt.xlabel('Network Depth (Number of Hidden Layers)')
        plt.ylabel('Average F1 Score')
        plt.title('Average F1 Score vs Network Depth (Adaptive Learning Rate)')
        plt.xticks(depths)
        plt.legend()
        plt.grid(True)
        os.makedirs(plots_output_folder, exist_ok=True)
        plt.savefig(os.path.join(plots_output_folder, f'f1_vs_network_depth__adaptive_lr_{question_part}.png'))
        plt.close()
        # plt.show()

        # Summary
        print("\nSummary of Results (Adaptive Learning Rate):")
        print("Hidden Layers | Depth | Train F1 | Test F1 | Train Accuracy | Test Accuracy")
        for config, depth, f1_train, f1_test, acc_train, acc_test in zip(hidden_layer_configs, depths, avg_f1_scores_train, avg_f1_scores_test, accuracy_train_list, accuracy_test_list):
            print(f"{str(config):<13} | {depth:<5} | {f1_train:.4f} | {f1_test:.4f} | {acc_train:.2f}% | {acc_test:.2f}%")

if __name__ == "__main__":
    main()