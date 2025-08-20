import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# CODE FOR PART (A) ==================================================================================================================================================================

# This is used to get the training 
def get_training_dataframe(filepath, target_column):
    df = pd.read_csv(filepath)
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    
    # (1) Rename the target column to 'y'
    df = df.rename(columns={target_column: 'y'})

    # (2) Get all attributes excluding the target column
    attributes = [col for col in df.columns if col != 'y']

    # (3) Determine if each attribute is numerical
    isNumAttribute = [pd.api.types.is_numeric_dtype(df[col]) for col in attributes]

    # (4) Create categorical mappings for non-numeric attributes
    categoricalMappings = {}
    for attr, is_num in zip(attributes, isNumAttribute):
        if not is_num:
            unique_vals = sorted(df[attr].unique())
            categoricalMappings[attr] = {val: idx for idx, val in enumerate(unique_vals)}

    # (5) Map 'y' to 0, 1, ..., k-1
    target_vals = sorted(df['y'].unique())
    targetMapping = {val: idx for idx, val in enumerate(target_vals)}
    df['y'] = df['y'].map(targetMapping)

    return df, attributes, isNumAttribute, categoricalMappings, targetMapping

def get_test_dataframe(filepath, target_column):
    df = pd.read_csv(filepath)
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Get the target column separately (if it exists)
    target = df[target_column] if target_column in df.columns else None

    # Drop the target column from the dataframe (if it exists)
    if target_column in df.columns:
        df = df.drop(columns=[target_column])

    return df, target

class Node:
    def __init__(self):
        self.numTrueSamples = 0
        self.numFalseSamples = 0
        self.isleaf = False

        self.splitAttribute = None
        self.splitValue = None # only useful in case of numerical attributes

        self.children = []
        self.depth = 0


class DecisionTree:
    def __init__(self, max_depth, attributes, isNumAttributes, categoricalMappings):
        self.root = Node() # Root node
        self.root.depth = 0

        # Catalogue 
        self.attributes = attributes
        self.isNumAttributes = isNumAttributes
        self.categoricalMappings = categoricalMappings

        self.max_depth = max_depth 

    def calculate_entropy(self, numTrue, numFalse):
        # This covers the case when either one of them is 0, both of them being 0 is not possible
        if(numTrue == 0 or numFalse == 0):
            return 0
        total = numTrue + numFalse
        probTrue = numTrue / total
        probFalse = numFalse / total
        entropy = - (probTrue * np.log2(probTrue) + probFalse * np.log2(probFalse))
        return entropy

    def calculate_information_gain(self, attribute, data):
        numTrue = np.sum(data['y'] == 1)
        numFalse = np.sum(data['y'] == 0)
        parentEntropy = self.calculate_entropy(numTrue, numFalse)

        total = len(data)

        attrIndex = self.attributes.index(attribute)
        attrValues = data[attribute].values

        childEntropy = 0
        if self.isNumAttributes[attrIndex]:
            # Numerical attribute
            median = np.median(attrValues)
            leftMask = attrValues <= median
            rightMask = attrValues > median

            yLeft = data['y'].values[leftMask]
            yRight = data['y'].values[rightMask]

            numTrueLeft = np.sum(yLeft == 1)
            numFalseLeft = np.sum(yLeft == 0)
            numTrueRight = np.sum(yRight == 1)
            numFalseRight = np.sum(yRight == 0)
            leftEntropy = self.calculate_entropy(numTrueLeft, numFalseLeft)
            rightEntropy = self.calculate_entropy(numTrueRight, numFalseRight)
            childEntropy = (len(yLeft) / total) * leftEntropy + (len(yRight) / total) * rightEntropy
        
        else:
            # Categorical attribute
            weighted_entropy = 0
            for category in self.categoricalMappings[attribute]:
                mask = attrValues == category
                ySubset = data['y'].values[mask]
                numTrue_cat = np.sum(ySubset == 1)
                numFalse_cat = len(ySubset) - numTrue_cat

                weight = len(ySubset) / total
                weighted_entropy += weight * self.calculate_entropy(numTrue_cat, numFalse_cat)
            
            childEntropy = weighted_entropy

        information_gain = parentEntropy - childEntropy
        return information_gain

    def best_attribute_to_split(self, data):
        bestGain = 0
        bestAttribute = None
        bestAttributeIndex = -1
        for i, attribute in enumerate(self.attributes):
            gain = self.calculate_information_gain(attribute, data)
            if gain > bestGain:
                bestGain = gain
                bestAttribute = attribute
                bestAttributeIndex = i
        
        if bestAttributeIndex == -1:
            return None, None
        
        if self.isNumAttributes[bestAttributeIndex]:
            # Numerical attribute
            median = np.median(data[bestAttribute].values)
            return bestAttribute, median
        else:
            # Categorical attribute
            return bestAttribute, None

    def build_tree(self, node, data):
        node.numTrueSamples = np.sum(data['y'] == 1)
        node.numFalseSamples = np.sum(data['y'] == 0)

        # Base Cases
        # (1) If the max depth is reached
        if node.depth >= self.max_depth:
            node.isleaf = True
            return

        # (2) If all samples are of the same class
        if node.numTrueSamples == 0 or node.numFalseSamples == 0:
            node.isleaf = True
            return

        # (3) If the best gain is 0 or less than that (i.e. best attribute to split is None)
        bestAttribute, medianValue = self.best_attribute_to_split(data)
        if bestAttribute is None:
            node.isleaf = True
            return

        node.splitAttribute = bestAttribute
        node.splitValue = medianValue
        node.isleaf = False

        attrIndex = self.attributes.index(bestAttribute)
        attrValues = data[bestAttribute].values

        if self.isNumAttributes[attrIndex]:
            # Numerical attribute => binary split
            leftMask = attrValues <= medianValue
            rightMask = attrValues > medianValue

            leftData = data[leftMask]
            rightData = data[rightMask]

            # This case would already be considered in Base Case (3), but let's keep it here
            if len(leftData) == 0 or len(rightData) == 0:
                node.isleaf = True
                return

            leftChild = Node()  
            rightChild = Node()

            leftChild.depth = node.depth + 1
            rightChild.depth = node.depth + 1

            node.children = [("<= {:.3f}".format(medianValue), leftChild),("> {:.3f}".format(medianValue), rightChild)]
            self.build_tree(leftChild, leftData)
            self.build_tree(rightChild, rightData)

        else:
            # Categorical attribute => multiple children
            node.children = []
            categories = list(self.categoricalMappings[bestAttribute].keys())
            for i in range(len(categories)):
                category = categories[i]
                mask = attrValues == category
                childData = data[mask]

                if len(childData) == 0:
                    continue

                childNode = Node()
                childNode.depth = node.depth + 1
                node.children.append((category,childNode))

                self.build_tree(childNode, childData)

    def predict_sample(self, row, node):
        if node.isleaf:
            return 1 if node.numTrueSamples > node.numFalseSamples else 0

        attrValue = row[node.splitAttribute]
        if self.isNumAttributes[self.attributes.index(node.splitAttribute)]:
            if attrValue <= node.splitValue:
                return self.predict_sample(row, node.children[0][1])
            else:
                return self.predict_sample(row, node.children[1][1])

        else:
            for category, childNode in node.children:
                if attrValue == category:
                    return self.predict_sample(row, childNode)

            return 1 if node.numTrueSamples > node.numFalseSamples else 0

    def predict(self, df):
        # Make predictions for each example
        predictions = [self.predict_sample(row, self.root) for _, row in df.iterrows()]
        return np.array(predictions)

    # Useful for debugging - to see the structure of the built tree
    def print_tree(self, node, indent=""):
        """Recursively prints the tree with all important fields."""
        # Print the important details of the node
        print(f"{indent}Node (Depth: {node.depth}):")
        print(f"{indent}  Is leaf: {node.isleaf}")
        print(f"{indent}  Num True Samples: {node.numTrueSamples}")
        print(f"{indent}  Num False Samples: {node.numFalseSamples}")
        
        if node.isleaf:
            print(f"{indent}  Leaf Node - No split")
        else:
            print(f"{indent}  Split on Attribute: {node.splitAttribute}")
            if node.splitValue is not None:
                print(f"{indent}  Split Value: {node.splitValue}")
            else:
                print(f"{indent}  Split on Categories: {self.categoricalMappings[node.splitAttribute]}")
        
        # Recursively print children (if any)
        if not node.isleaf:
            for category, childNode in node.children:
                print(f"{indent}  -- Go to category: {category}")
                self.print_tree(childNode, indent + "    ")

# CODE FOR PART (B) AND PART (C) ===================================================================================================================================================================

def get_training_data_OHE(filepath, target_column):
    df = pd.read_csv(filepath)
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # (1) Rename the target column to 'y'
    df = df.rename(columns={target_column: 'y'})

    # (2) Map 'y' to 0, 1, ..., k-1
    target_vals = sorted(df['y'].unique())
    targetMapping = {val: idx for idx, val in enumerate(target_vals)}
    df['y'] = df['y'].map(targetMapping)

    # (3) Separate features and apply One-Hot Encoding to categorical features
    feature_df = df.drop(columns=['y'])
    isNumAttribute_original = [pd.api.types.is_numeric_dtype(feature_df[col]) for col in feature_df.columns]

    num_cols = [col for col, is_num in zip(feature_df.columns, isNumAttribute_original) if is_num]
    cat_cols = [col for col, is_num in zip(feature_df.columns, isNumAttribute_original) if not is_num]

    # One-hot encode categorical columns
    df_encoded = pd.get_dummies(feature_df, columns=cat_cols, prefix=cat_cols).astype(int)

    # (4) Update the list of attributes and isNumAttribute
    attributes = list(df_encoded.columns)
    isNumAttribute = []
    for col in attributes:
        # If it was a numerical column before encoding, it's still numerical
        if col in num_cols:
            isNumAttribute.append(True)
        else:
            # All one-hot columns are from categorical columns
            isNumAttribute.append(False)

    # (5) Add the target column back
    df_encoded['y'] = df['y']

    return df_encoded, attributes, isNumAttribute, targetMapping

def get_test_data_OHE(filepath, target_column, attributes):
    df = pd.read_csv(filepath)
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Extract and drop the target if it exists
    target = df[target_column] if target_column in df.columns else None
    if target_column in df.columns:
        df = df.drop(columns=[target_column])

    # Identify numeric and categorical columns in original test data
    isNumAttribute_original = [pd.api.types.is_numeric_dtype(df[col]) for col in df.columns]
    num_cols = [col for col, is_num in zip(df.columns, isNumAttribute_original) if is_num]
    cat_cols = [col for col, is_num in zip(df.columns, isNumAttribute_original) if not is_num]

    # One-hot encode test data
    df_encoded = pd.get_dummies(df, columns=cat_cols, prefix=cat_cols).astype(int)

    # Add missing columns (those present in training but missing in test)
    for col in attributes:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Reorder columns to match training attributes
    df_encoded = df_encoded[attributes]

    return df_encoded, target

class NodeOHE:
    def __init__(self):
        self.numTrueSamples = 0
        self.numFalseSamples = 0
        self.isleaf = False

        self.splitAttribute = None
        self.splitValue = None # only useful in case of numerical attributes

        self.leftChild = None
        self.rightChild = None
        self.depth = 0

        # New fields for pruning
        self.parent = None           # Pointer to parent node
        self.numValidSamples = 0     # Validation samples reaching this node
        self.numValidTrue = 0        # Validation samples with y=1
        self.numValidFalse = 0       # Validation samples with y=0
        self.numValidCorrect = 0     # Validation samples correctly classified

class DecisionTreeOHE:
    def __init__(self, max_depth, attributes, isNumAttributes):
        self.root = NodeOHE() # Root node
        self.root.depth = 0

        # Catalogue 
        self.attributes = attributes
        self.isNumAttributes = isNumAttributes

        self.max_depth = max_depth 

        # Prune Tracking
        self.num_nodes_list = []
        self.train_acc_list = []
        self.valid_acc_list = []
        self.test_acc_list = []

    def calculate_entropy(self, numTrue, numFalse):
        # This covers the case when either one of them is 0, both of them being 0 is not possible
        if(numTrue == 0 or numFalse == 0):
            return 0
        total = numTrue + numFalse
        probTrue = numTrue / total
        probFalse = numFalse / total
        entropy = - (probTrue * np.log2(probTrue) + probFalse * np.log2(probFalse))
        return entropy
    
    def calculate_information_gain(self, attribute, data):
        numTrue = np.sum(data['y'] == 1)
        numFalse = np.sum(data['y'] == 0)
        parentEntropy = self.calculate_entropy(numTrue, numFalse)

        total = len(data)

        attrIndex = self.attributes.index(attribute)
        attrValues = data[attribute].values

        if self.isNumAttributes[attrIndex]:
            # Numerical attribute
            median = np.median(attrValues)
            leftMask = attrValues <= median

            yLeft = data['y'].values[leftMask]
            yRight = data['y'].values[~leftMask]

        else:
            leftMask = attrValues == 0
            yLeft = data['y'].values[leftMask]
            yRight = data['y'].values[~leftMask]

        numTrueLeft = np.sum(yLeft == 1)
        numFalseLeft = np.sum(yLeft == 0)
        numTrueRight = np.sum(yRight == 1)
        numFalseRight = np.sum(yRight == 0)
        leftEntropy = self.calculate_entropy(numTrueLeft, numFalseLeft)
        rightEntropy = self.calculate_entropy(numTrueRight, numFalseRight)
        childEntropy = (len(yLeft) / total) * leftEntropy + (len(yRight) / total) * rightEntropy
        informationGain = parentEntropy - childEntropy
        
        return informationGain

    def best_attribute_to_split(self, data):
        bestGain = 0
        bestAttribute = None
        bestAttributeIndex = -1
        for i, attribute in enumerate(self.attributes):
            gain = self.calculate_information_gain(attribute, data)
            if gain > bestGain:
                bestGain = gain
                bestAttribute = attribute
                bestAttributeIndex = i
        
        if bestAttributeIndex == -1:
            return None, None
        
        if self.isNumAttributes[bestAttributeIndex]:
            # Numerical attribute
            median = np.median(data[bestAttribute].values)
            return bestAttribute, median
        else:
            # Categorical attribute
            return bestAttribute, None
        
    def build_tree(self, data, node):
        node.numTrueSamples = np.sum(data['y'] == 1)
        node.numFalseSamples = np.sum(data['y'] == 0)

        # Base Cases
        # (1) If the max depth is reached
        if node.depth >= self.max_depth:
            node.isleaf = True
            return

        # (2) If all samples are of the same class
        if node.numTrueSamples == 0 or node.numFalseSamples == 0:
            node.isleaf = True
            return

        # (3) If the best gain is 0 or less than that (i.e. best attribute to split is None)
        bestAttribute, medianValue = self.best_attribute_to_split(data)
        if bestAttribute is None:
            node.isleaf = True
            return
        
        node.splitAttribute = bestAttribute
        node.splitValue = medianValue
        node.isleaf = False

        attrIndex = self.attributes.index(bestAttribute)
        attrValues = data[bestAttribute].values

        if self.isNumAttributes[attrIndex]:
            # Numerical attribute
            leftMask = attrValues <= medianValue
            # Split data efficiently using boolean masks
            leftData = data[leftMask]
            rightData = data[~leftMask]
            node.splitValue = medianValue
        
        else:
            leftMask = attrValues == 0
            # Split data efficiently using boolean masks
            leftData = data[leftMask]
            rightData = data[~leftMask]

        if len(leftData) == 0 or len(rightData) == 0:
            node.isleaf = True
            return
        
        node.leftChild = NodeOHE()
        node.leftChild.depth = node.depth + 1
        node.leftChild.parent = node  # Set parent
        self.build_tree(leftData, node.leftChild)

        node.rightChild = NodeOHE()
        node.rightChild.depth = node.depth + 1
        node.rightChild.parent = node  # Set parent
        self.build_tree(rightData, node.rightChild)

    def predict_sample(self, row, node):
        if node.isleaf:
            return 1 if node.numTrueSamples > node.numFalseSamples else 0

        attrIndex = self.attributes.index(node.splitAttribute)
        attrValue = row[node.splitAttribute]

        if self.isNumAttributes[attrIndex]:
            # Numerical attribute
            if attrValue <= node.splitValue:
                return self.predict_sample(row, node.leftChild)
            else:
                return self.predict_sample(row, node.rightChild)
        else:
            # Categorical attribute
            if attrValue == 0:
                return self.predict_sample(row, node.leftChild)
            else:
                return self.predict_sample(row, node.rightChild)
            
    def predict(self, data):
        # Handle missing columns in test data
        missing_cols = set(self.attributes) - set(data.columns)
        for col in missing_cols:
            data[col] = 0  # Add missing column with 0s
        
        # Only use columns that were in training data
        relevant_cols = [col for col in data.columns if col in self.attributes or col == 'y']
        data_subset = data[relevant_cols]
        
        predictions = [self.predict_sample(row, self.root) for _, row in data_subset.iterrows()]
        return np.array(predictions)
    
    def update_validation_statistics(self, row, y, node):
        node.numValidSamples += 1
        if y == 1:
            node.numValidTrue += 1
        else:
            node.numValidFalse += 1
        if node.isleaf:
            prediction = 1 if node.numTrueSamples > node.numFalseSamples else 0
            return prediction == y
        attrIndex = self.attributes.index(node.splitAttribute)
        attrValue = row[node.splitAttribute]
        if self.isNumAttributes[attrIndex]:
            if attrValue <= node.splitValue:
                is_correct = self.update_validation_statistics(row, y, node.leftChild)
            else:
                is_correct = self.update_validation_statistics(row, y, node.rightChild)
        else:
            if attrValue == 0:
                is_correct = self.update_validation_statistics(row, y, node.leftChild)
            else:
                is_correct = self.update_validation_statistics(row, y, node.rightChild)
        if is_correct:
            node.numValidCorrect += 1
        return is_correct
    
    def _get_internal_nodes(self, node):
        if node is None or node.isleaf:
            return []
        return [node] + self._get_internal_nodes(node.leftChild) + self._get_internal_nodes(node.rightChild)
    
    def count_nodes(self, node=None):
        if node is None:
            node = self.root
        if node.isleaf:
            return 1
        return 1 + self.count_nodes(node.leftChild) + self.count_nodes(node.rightChild)
    
    def train_valid_acc(self, df):
        predictions = self.predict(df)
        true_labels = df['y'].values
        return np.mean(predictions == true_labels)
    
    def test_acc(self, test_df, test_target):
        predictions = self.predict(test_df)
        true_labels = test_target.values
        return np.mean(predictions == true_labels)
    
    def prune(self, valid_df, train_df, test_df, test_target):
        # Step 1: Compute initial validation statistics
        for _, row in valid_df.iterrows():
            self.update_validation_statistics(row, row['y'], self.root)
        
        # Step 2: Collect all internal nodes
        internal_nodes = self._get_internal_nodes(self.root)
        
        # Step 3: Iteratively prune nodes
        while internal_nodes:
            # Tracking step
            self.num_nodes_list.append(self.count_nodes())
            self.train_acc_list.append(self.train_valid_acc(train_df))
            self.valid_acc_list.append(self.train_valid_acc(valid_df))
            self.test_acc_list.append(self.test_acc(test_df, test_target))

            best_delta = -1
            best_node = None
            # Evaluate pruning each internal node
            for node in internal_nodes:
                # Majority class based on training data
                M_N = 1 if node.numTrueSamples > node.numFalseSamples else 0
                # New correct predictions if pruned
                new_correct = node.numValidTrue if M_N == 1 else node.numValidFalse
                # Change in correct predictions
                delta = new_correct - node.numValidCorrect
                if delta > best_delta:
                    best_delta = delta
                    best_node = node
            
            # If no improvement, stop pruning
            if best_delta <= 0:
                break
            
            # Prune the best node
            nodes_to_remove = self._get_internal_nodes(best_node)
            best_node.isleaf = True
            best_node.leftChild = None
            best_node.rightChild = None

            # Remove the child nodes from the list of internal nodes
            for node in nodes_to_remove:
                if node in internal_nodes:
                    internal_nodes.remove(node)
            
            # Update ancestors' numValidCorrect
            current = best_node.parent
            while current is not None:
                current.numValidCorrect += best_delta
                current = current.parent

# CODE FOR PART (D) AND PART (E) ======================================================================================================================================================
# Function to load and preprocess data
def preprocess_data(file_path):
    # Load data
    df = pd.read_csv(file_path)
    
    # Strip whitespace from string columns
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip()
    
    # Separate features and target
    X = df.drop('income', axis=1)
    y = df['income'].apply(lambda x: 1 if x == '>50K' else 0)
    
    # Identify numerical and categorical columns
    numerical_cols = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
    categorical_cols = [col for col in X.columns if col not in numerical_cols]
    
    # Create a new dataframe with just numerical columns
    X_numerical = X[numerical_cols].reset_index(drop=True)
    
    # One-hot encode categorical columns using pandas get_dummies
    X_categorical = pd.get_dummies(X[categorical_cols], prefix_sep='_', dummy_na=False)
    
    # Combine numerical and categorical features
    X_processed = pd.concat([X_numerical, X_categorical], axis=1)
    
    # Store category values for consistent encoding of test/validation sets
    category_mappings = {}
    for col in categorical_cols:
        category_mappings[col] = list(X[col].unique())
    
    return X_processed, y, category_mappings, numerical_cols, categorical_cols

# Function to process test and validation data with consistent encoding
def process_test_valid_data(file_path, category_mappings, numerical_cols, categorical_cols):
    # Load data
    df = pd.read_csv(file_path)
    
    # Strip whitespace from string columns
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip()
    
    if 'income' in df.columns:
        y = df['income'].apply(lambda x: 1 if x == '>50K' else 0)
        X = df.drop('income', axis=1)
    else:
        y = None
        X = df
    
    # Create a new dataframe with just numerical columns
    X_numerical = X[numerical_cols].reset_index(drop=True)
    
    # Initialize an empty DataFrame for categorical features
    X_categorical_encoded = pd.DataFrame()
    
    # Process each categorical column individually to ensure consistent encoding
    for col in categorical_cols:
        # Create a temporary series with categories from training set
        temp_series = pd.Series(X[col])
        
        # Create a DataFrame with one-hot encoding for this column
        dummies = pd.get_dummies(temp_series, prefix=col, prefix_sep='_')
        
        # Add dummy columns that are in training set but not in this dataset
        for category in category_mappings[col]:
            dummy_col = f"{col}_{category}"
            if dummy_col not in dummies.columns:
                dummies[dummy_col] = 0
        
        # Remove columns that weren't in the training set
        cols_to_keep = [f"{col}_{cat}" for cat in category_mappings[col] if f"{col}_{cat}" in dummies.columns]
        dummies = dummies[cols_to_keep]
        
        # Add to our encoded categorical dataframe
        X_categorical_encoded = pd.concat([X_categorical_encoded, dummies], axis=1)
    
    # Combine with numerical features
    X_processed = pd.concat([X_numerical, X_categorical_encoded], axis=1)
    
    # Ensure all columns from training are present in the same order
    return X_processed, y

# Custom scorer using OOB score (GridSearchCV doesn’t natively support it)
def oob_scorer(estimator, X, y):
            return estimator.oob_score_

# Since we're diretly using libraries, most of the code for this part is in main()

# MAIN CODE ===========================================================================================================================================================================
def main():
    if len(sys.argv) != 6:
        sys.exit(1)

    train_path = sys.argv[1]
    valid_path = sys.argv[2]
    test_path = sys.argv[3]
    output_folder = sys.argv[4]
    question_part = sys.argv[5]

    target_column = 'income' 
    plots_output_folder = '../../output/Q1_plots'

    if(question_part == 'a'):
        train_df, train_attributes, train_isNumAttributes, train_categoricalMappings, train_TargetMapping = get_training_dataframe(train_path, target_column)
        test_df, test_target = get_test_dataframe(test_path, target_column)
        test_target = test_target.map(train_TargetMapping)

        max_depths = [5, 10, 15, 20]
        train_accuracies = []
        test_accuracies = []

        for depth in max_depths:
            dt = DecisionTree(max_depth=depth, attributes=train_attributes, isNumAttributes=train_isNumAttributes, categoricalMappings=train_categoricalMappings)
            dt.build_tree(dt.root, train_df)

            train_predictions = dt.predict(train_df)
            train_accuracy = np.mean(train_predictions == train_df['y'].values)

            test_predictions = dt.predict(test_df)
            test_accuracy = np.mean(test_predictions == test_target.values)

            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)

            print(f"Max Depth: {depth} | Train Accuracy: {train_accuracy:.4f} | Test Accuracy: {test_accuracy:.4f}")

            if depth == 20:
                inverse_target_mapping = {0 : '<=50K', 1 : '>50K'}
                mapped_predictions = [inverse_target_mapping[pred] for pred in test_predictions]

                prediction_df = pd.DataFrame({'predictions': mapped_predictions})
                os.makedirs(output_folder, exist_ok=True)
                prediction_df.to_csv(os.path.join(output_folder, f'prediction_{question_part}.csv'), index=False)

        # Plotting
        plt.figure(figsize=(10,6))
        plt.plot(max_depths, train_accuracies, marker='o', label='Train Accuracy')
        plt.plot(max_depths, test_accuracies, marker='s', label='Test Accuracy')
        plt.title('Train and Test Accuracy vs. Tree Depth')
        plt.xlabel('Maximum Depth')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        os.makedirs(plots_output_folder, exist_ok=True)
        plt.savefig(os.path.join(plots_output_folder, f'accuracy_vs_depth_{question_part}.png'))
        plt.close()

    elif(question_part == 'b'):
        train_df, train_attributes, train_isNumAttributes, train_TargetMapping = get_training_data_OHE(train_path, target_column)
        test_df, test_target = get_test_data_OHE(test_path, target_column, train_attributes)
        test_target = test_target.map(train_TargetMapping)
        max_depths = [5, 10, 15, 20, 25, 35, 45, 55]
        train_accuracies = []
        test_accuracies = []

        for depth in max_depths:
            dt = DecisionTreeOHE(max_depth=depth, attributes=train_attributes, isNumAttributes=train_isNumAttributes)
            dt.build_tree(train_df, dt.root)

            train_predictions = dt.predict(train_df)
            train_accuracy = np.mean(train_predictions == train_df['y'].values)

            test_predictions = dt.predict(test_df)
            test_accuracy = np.mean(test_predictions == test_target.values)

            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)

            print(f"Max Depth: {depth} | Train Accuracy: {train_accuracy:.4f} | Test Accuracy: {test_accuracy:.4f}")

            if depth == 55:
                inverse_target_mapping = {0 : '<=50K', 1 : '>50K'}
                mapped_predictions = [inverse_target_mapping[pred] for pred in test_predictions]

                prediction_df = pd.DataFrame({'predictions': mapped_predictions})
                os.makedirs(output_folder, exist_ok=True)
                prediction_df.to_csv(os.path.join(output_folder, f'prediction_{question_part}.csv'), index=False)

        # Plotting
        plt.figure(figsize=(10,6))
        plt.plot(max_depths, train_accuracies, marker='o', label='Train Accuracy')
        plt.plot(max_depths, test_accuracies, marker='s', label='Test Accuracy')
        plt.title('Train and Test Accuracy vs. Tree Depth')
        plt.xlabel('Maximum Depth')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        os.makedirs(plots_output_folder, exist_ok=True)
        plt.savefig(os.path.join(plots_output_folder, f'accuracy_vs_depth_{question_part}.png'))
        plt.close()

    elif(question_part == 'c'):
        train_df, train_attributes, train_isNumAttributes, train_TargetMapping = get_training_data_OHE(train_path, target_column)
        valid_df, _, _, _ = get_training_data_OHE(valid_path, target_column)
        test_df, test_target = get_test_data_OHE(test_path, target_column, train_attributes)
        test_target = test_target.map(train_TargetMapping)

        max_depths = [25, 35, 45, 55]
        train_accuracies = []
        valid_accuracies = []
        test_accuracies = []

        for depth in max_depths:
            # Initialize and train the decision tree
            dt = DecisionTreeOHE(max_depth=depth, attributes=train_attributes, isNumAttributes=train_isNumAttributes)
            dt.build_tree(train_df, dt.root)

            # Prune the tree (corrected argument order)
            dt.prune(valid_df, train_df, test_df, test_target)

            # Compute accuracies after pruning
            train_predictions = dt.predict(train_df)
            train_accuracy = np.mean(train_predictions == train_df['y'].values)

            valid_predictions = dt.predict(valid_df)
            valid_accuracy = np.mean(valid_predictions == valid_df['y'].values)

            test_predictions = dt.predict(test_df)
            test_accuracy = np.mean(test_predictions == test_target.values)

            train_accuracies.append(train_accuracy)
            valid_accuracies.append(valid_accuracy)
            test_accuracies.append(test_accuracy)

            print(f"Max Depth: {depth} | Train Accuracy: {train_accuracy:.4f} | Valid Accuracy: {valid_accuracy:.4f} | Test Accuracy: {test_accuracy:.4f}")

            # Plot accuracies vs. number of nodes during pruning
            plt.figure(figsize=(10, 6))
            plt.plot(dt.num_nodes_list, dt.train_acc_list, marker='o', label='Train Accuracy')
            plt.plot(dt.num_nodes_list, dt.valid_acc_list, marker='^', label='Validation Accuracy')
            plt.plot(dt.num_nodes_list, dt.test_acc_list, marker='s', label='Test Accuracy')
            plt.title(f'Accuracy vs. Number of Nodes (Max Depth {depth})')
            plt.xlabel('Number of Nodes')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
            os.makedirs(plots_output_folder, exist_ok=True)
            plt.savefig(os.path.join(plots_output_folder, f'accuracy_vs_num_nodes_{question_part}_{depth}.png'))
            plt.close()

            # Save predictions for max_depth=55
            if depth == 55:
                inverse_target_mapping = {0 : '<=50K', 1 : '>50K'}
                mapped_predictions = [inverse_target_mapping[pred] for pred in test_predictions]

                prediction_df = pd.DataFrame({'predictions': mapped_predictions})
                os.makedirs(output_folder, exist_ok=True)
                prediction_df.to_csv(os.path.join(output_folder, f'prediction_{question_part}.csv'), index=False)

        # Plot accuracies vs. max depth
        plt.figure(figsize=(10, 6))
        plt.plot(max_depths, train_accuracies, marker='o', label='Train Accuracy')
        plt.plot(max_depths, valid_accuracies, marker='^', label='Validation Accuracy')
        plt.plot(max_depths, test_accuracies, marker='s', label='Test Accuracy')
        plt.title('Accuracy vs. Tree Depth')
        plt.xlabel('Maximum Depth')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        os.makedirs(plots_output_folder, exist_ok=True)
        plt.savefig(os.path.join(plots_output_folder, f'accuracy_vs_depth_{question_part}.png'))
        plt.close()

    elif(question_part == 'd'):
        np.random.seed(42)
        # Load and preprocess training data
        X_train, y_train, category_mappings, numerical_cols, categorical_cols = preprocess_data('../../data/Q1/train.csv')

        # Load and preprocess validation and test data
        X_valid, y_valid = process_test_valid_data('../../data/Q1/valid.csv', category_mappings, numerical_cols, categorical_cols)
        X_test, y_test = process_test_valid_data('../../data/Q1/test.csv', category_mappings, numerical_cols, categorical_cols)

        # Verify all datasets have the same features
        print(f"Training set shape: {X_train.shape}")
        print(f"Validation set shape: {X_valid.shape}")
        print(f"Test set shape: {X_test.shape}")

        # Ensure validation and test dataframes have the same columns as training
        X_valid = X_valid.reindex(columns=X_train.columns, fill_value=0)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        # Verify again after alignment
        print(f"After alignment - Training set shape: {X_train.shape}")
        print(f"After alignment - Validation set shape: {X_valid.shape}")
        print(f"After alignment - Test set shape: {X_test.shape}")

        # Part (i): Varying max_depth
        depth_values = [25, 35, 45, 55]
        train_acc_depth = []
        valid_acc_depth = []
        test_acc_depth = []

        plt.figure(figsize=(12, 6))

        for depth in depth_values:
            # Create and train the Decision Tree model
            dt_classifier = DecisionTreeClassifier(criterion='entropy', max_depth=depth, random_state=42)
            dt_classifier.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = dt_classifier.predict(X_train)
            y_valid_pred = dt_classifier.predict(X_valid)
            y_test_pred = dt_classifier.predict(X_test)
            
            # Calculate accuracy
            train_accuracy = accuracy_score(y_train, y_train_pred)
            valid_accuracy = accuracy_score(y_valid, y_valid_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            
            # Store accuracies
            train_acc_depth.append(train_accuracy)
            valid_acc_depth.append(valid_accuracy)
            test_acc_depth.append(test_accuracy)
            
            print(f"max_depth={depth}, Train Accuracy: {train_accuracy:.4f}, Valid Accuracy: {valid_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

        # Plot accuracies vs. max_depth
        plt.subplot(1, 2, 1)
        plt.plot(depth_values, train_acc_depth, 'o-', label='Train')
        plt.plot(depth_values, valid_acc_depth, 's-', label='Validation')
        plt.plot(depth_values, test_acc_depth, '^-', label='Test')
        plt.xlabel('Max Depth')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs. Max Depth')
        plt.ylim(0.5, 1.1)  # Set y-axis limits from 0 to 1
        plt.grid(True)
        plt.legend()

        # Find best max_depth based on validation accuracy
        best_depth_idx = np.argmax(valid_acc_depth)
        best_depth = depth_values[best_depth_idx]
        print(f"\nBest max_depth value: {best_depth}")
        print(f"Best validation accuracy: {valid_acc_depth[best_depth_idx]:.4f}")
        print(f"Corresponding test accuracy: {test_acc_depth[best_depth_idx]:.4f}")

        # Part (ii): Varying ccp_alpha (pruning parameter)
        alpha_values = [0.001, 0.01, 0.1, 0.2]
        train_acc_alpha = []
        valid_acc_alpha = []
        test_acc_alpha = []

        for alpha in alpha_values:
            # Create and train the Decision Tree model
            dt_classifier = DecisionTreeClassifier(criterion='entropy', ccp_alpha=alpha, random_state=42, max_depth=55)
            dt_classifier.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = dt_classifier.predict(X_train)
            y_valid_pred = dt_classifier.predict(X_valid)
            y_test_pred = dt_classifier.predict(X_test)
            
            # Calculate accuracy
            train_accuracy = accuracy_score(y_train, y_train_pred)
            valid_accuracy = accuracy_score(y_valid, y_valid_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            
            # Store accuracies
            train_acc_alpha.append(train_accuracy)
            valid_acc_alpha.append(valid_accuracy)
            test_acc_alpha.append(test_accuracy)
            
            print(f"ccp_alpha={alpha}, Train Accuracy: {train_accuracy:.4f}, Valid Accuracy: {valid_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

        # Plot accuracies vs. ccp_alpha
        plt.subplot(1, 2, 2)
        plt.plot(alpha_values, train_acc_alpha, 'o-', label='Train')
        plt.plot(alpha_values, valid_acc_alpha, 's-', label='Validation')
        plt.plot(alpha_values, test_acc_alpha, '^-', label='Test')
        plt.xlabel('ccp_alpha')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs. ccp_alpha')
        plt.ylim(0.5, 1.1)  # Set y-axis limits from 0 to 1
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        os.makedirs(plots_output_folder, exist_ok=True)
        plt.savefig(os.path.join(plots_output_folder, f'decision_tree_param_tuning_{question_part}.png'))
        # plt.show()

        # Find best ccp_alpha based on validation accuracy
        best_alpha_idx = np.argmax(valid_acc_alpha)
        best_alpha = alpha_values[best_alpha_idx]
        print(f"\nBest ccp_alpha value: {best_alpha}")
        print(f"Best validation accuracy: {valid_acc_alpha[best_alpha_idx]:.4f}")
        print(f"Corresponding test accuracy: {test_acc_alpha[best_alpha_idx]:.4f}")

        # Compare the best models from (i) and (ii)
        print("\nComparison of best models:")
        print(f"Best max_depth model: Validation Accuracy = {valid_acc_depth[best_depth_idx]:.4f}, Test Accuracy = {test_acc_depth[best_depth_idx]:.4f}")
        print(f"Best ccp_alpha model: Validation Accuracy = {valid_acc_alpha[best_alpha_idx]:.4f}, Test Accuracy = {test_acc_alpha[best_alpha_idx]:.4f}")

        # Train final model with best parameters
        if valid_acc_depth[best_depth_idx] >= valid_acc_alpha[best_alpha_idx]:
            print(f"\nFinal model uses max_depth={best_depth}")
            final_model = DecisionTreeClassifier(criterion='entropy', max_depth=best_depth, random_state=42)
        else:
            print(f"\nFinal model uses ccp_alpha={best_alpha}")
            final_model = DecisionTreeClassifier(criterion='entropy', ccp_alpha=best_alpha, random_state=42)

        final_model.fit(X_train, y_train)
        final_train_acc = accuracy_score(y_train, final_model.predict(X_train))
        final_valid_acc = accuracy_score(y_valid, final_model.predict(X_valid))
        final_test_acc = accuracy_score(y_test, final_model.predict(X_test))

        inverse_target_mapping = {0 : '<=50K', 1 : '>50K'}
        mapped_predictions = [inverse_target_mapping[pred] for pred in final_model.predict(X_test)]
        prediction_df = pd.DataFrame({'predictions': mapped_predictions})

        print(f"Final model results:")
        print(f"Training accuracy: {final_train_acc:.4f}")
        print(f"Validation accuracy: {final_valid_acc:.4f}")
        print(f"Test accuracy: {final_test_acc:.4f}")
        os.makedirs(output_folder, exist_ok=True)
        prediction_df.to_csv(os.path.join(output_folder, f'prediction_{question_part}.csv'), index=False)

    elif(question_part == 'e'):
        np.random.seed(42)
        # Load and preprocess training data
        X_train, y_train, category_mappings, numerical_cols, categorical_cols = preprocess_data('../../data/Q1/train.csv')

        # Load and preprocess validation and test data
        X_valid, y_valid = process_test_valid_data('../../data/Q1/valid.csv', category_mappings, numerical_cols, categorical_cols)
        X_test, y_test = process_test_valid_data('../../data/Q1/test.csv', category_mappings, numerical_cols, categorical_cols)

        # Verify all datasets have the same features
        print(f"Training set shape: {X_train.shape}")
        print(f"Validation set shape: {X_valid.shape}")
        print(f"Test set shape: {X_test.shape}")

        # Ensure validation and test dataframes have the same columns as training
        X_valid = X_valid.reindex(columns=X_train.columns, fill_value=0)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        # Verify again after alignment
        print(f"After alignment - Training set shape: {X_train.shape}")
        print(f"After alignment - Validation set shape: {X_valid.shape}")
        print(f"After alignment - Test set shape: {X_test.shape}")
        param_grid = {
            'n_estimators': [50, 150, 250, 350],
            'max_features': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
            'min_samples_split': [2, 4, 6, 8, 10],
        }

        # Create base estimator
        rf_base = RandomForestClassifier(
            criterion='entropy',
            oob_score=True,
            random_state=42,
            n_jobs=-1,
            bootstrap=True
        )

        # Use GridSearchCV with custom scorer
        grid_search = GridSearchCV(
            estimator=rf_base,
            param_grid=param_grid,
            cv=2, # Use all data for OOB - Cross-Validation is useless here since OOB does a similar job : using 5 for now - default(need to figure this out later)
            scoring=oob_scorer,
            verbose=3,
            n_jobs=-1  # Parallelizes across parameter combinations
        )

        # Fit grid search
        grid_search.fit(X_train, y_train)

        # Best model and parameters
        best_rf = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_oob_accuracy = grid_search.best_score_

        # Accuracy evaluations
        train_accuracy = accuracy_score(y_train, best_rf.predict(X_train))
        valid_accuracy = accuracy_score(y_valid, best_rf.predict(X_valid))
        test_accuracy = accuracy_score(y_test, best_rf.predict(X_test))

        inverse_target_mapping = {0 : '<=50K', 1 : '>50K'}
        mapped_predictions = [inverse_target_mapping[pred] for pred in best_rf.predict(X_test)]
        prediction_df = pd.DataFrame({'predictions': mapped_predictions})

        # Print results
        print(f"\nBest Parameters: {best_params}")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"OOB Accuracy: {best_oob_accuracy:.4f}")
        print(f"Validation Accuracy: {valid_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        os.makedirs(output_folder, exist_ok=True)
        prediction_df.to_csv(os.path.join(output_folder, f'prediction_{question_part}.csv'), index=False)

if __name__ == "__main__":
    main()