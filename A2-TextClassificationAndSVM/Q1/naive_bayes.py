import numpy as np
import pandas as pd

class NaiveBayes:
    def __init__(self):
        # Vocabulary of the training data and the number of classes
        self.vocabulary = set()
        self.classes = []
        # Prior probabilities of each class (phi_y)
        self.phi_y = {}
        # Conditional probabilities of each token given each class (phi_x|y, same for every j)
        self.phi_x_y = {}
        # Total tokens per class - need it in the predict function, avoids recalculation
        self.total_tokens_y = {}
        # Smoothening parameter
        self.smoothening = 1 # gets update when fit() is called

        # Additional storage for the new fit_dual and predict_dual functions
        self.vocabulary_title = set()
        self.vocabulary_desc = set()
        self.phi_x_y_title = {}
        self.phi_x_y_desc = {}
        self.total_tokens_y_desc = {}
        self.total_tokens_y_title = {}
        
    def fit(self, df, smoothening, class_col = "Class Index", text_col = "Tokenized Description"):
        # Extract classes
        self.classes = df[class_col].unique()

        # Get the class counts and the total number of samples
        class_counts = df[class_col].value_counts()
        m = len(df)

        # Calculate the prior probabilities of each class
        for c in self.classes:
            self.phi_y[c] = np.log(class_counts[c] / m)

        # Build vocabulary, calculate token frequencies per class
        num_tokens_per_class = {c: {} for c in self.classes}
        total_tokens_per_class = {c: 0 for c in self.classes}

        for _, row in df.iterrows():
            c = row[class_col]
            tokens = row[text_col]
            for token in tokens:
                self.vocabulary.add(token)
                if token not in num_tokens_per_class[c]:
                    num_tokens_per_class[c][token] = 0
                num_tokens_per_class[c][token] += 1
                total_tokens_per_class[c] += 1

        self.total_tokens_y = total_tokens_per_class
        self.smoothening = smoothening

        # Calculate the conditional probabilities of each token given each class
        v = len(self.vocabulary)
        for c in self.classes:
            self.phi_x_y[c] = {token: np.log((num_tokens_per_class[c].get(token, 0) + smoothening) / (total_tokens_per_class[c] + smoothening * v)) for token in self.vocabulary}
        
    
    def predict(self, df, text_col = "Tokenized Description", predicted_col = "Predicted"):
        v = len(self.vocabulary)
        k = len(self.classes)

        # I actually have a lot of doubts in this, but let's go with this for now
        # Doubt: Should I update the |V| ? I think I should, but I'm not sure (but OOV tokens should be handled)
        # Also, should it be (|V| + k) in denominator or (|V| + num_tokens in class)
        def predict_row(row):
            tokens = row[text_col]
            class_scores = {c: self.phi_y[c] for c in self.classes}
            for c in self.classes:
                for token in tokens:
                    if token in self.vocabulary:
                        class_scores[c] += self.phi_x_y[c].get(token, 0)
                    else:
                        class_scores[c] += np.log(self.smoothening / (self.total_tokens_y[c] + self.smoothening * v))
            return max(class_scores, key = class_scores.get)

        df[predicted_col] = df.apply(predict_row, axis = 1)     

    # def fit_dual(self, df, smoothening, class_col="Class Index", title_col="Tokenized Title", desc_col="Tokenized Description"):
    #     self.classes = df[class_col].unique()
    #     class_counts = df[class_col].value_counts()
    #     m = len(df)
        
    #     for c in self.classes:
    #         self.phi_y[c] = np.log(class_counts[c] / m)
        
    #     num_tokens_title = {c: {} for c in self.classes}
    #     num_tokens_desc = {c: {} for c in self.classes}
    #     total_tokens_title = {c: 0 for c in self.classes}
    #     total_tokens_desc = {c: 0 for c in self.classes}
        
    #     for _, row in df.iterrows():
    #         c = row[class_col]
    #         title_tokens = row[title_col]
    #         desc_tokens = row[desc_col]
            
    #         for token in title_tokens:
    #             self.vocabulary_title.add(token)
    #             num_tokens_title[c][token] = num_tokens_title[c].get(token, 0) + 1
    #             total_tokens_title[c] += 1
            
    #         for token in desc_tokens:
    #             self.vocabulary_desc.add(token)
    #             num_tokens_desc[c][token] = num_tokens_desc[c].get(token, 0) + 1
    #             total_tokens_desc[c] += 1

    #     self.total_tokens_y_title = total_tokens_title
    #     self.total_tokens_y_desc = total_tokens_desc
    #     self.smoothening = smoothening
        
    #     v_title = len(self.vocabulary_title)
    #     v_desc = len(self.vocabulary_desc)
        
    #     for c in self.classes:
    #         self.phi_x_y_title[c] = {token: np.log((num_tokens_title[c].get(token, 0) + smoothening) / (total_tokens_title[c] + smoothening * v_title)) for token in self.vocabulary_title}
    #         self.phi_x_y_desc[c] = {token: np.log((num_tokens_desc[c].get(token, 0) + smoothening) / (total_tokens_desc[c] + smoothening * v_desc)) for token in self.vocabulary_desc}


    # def predict_dual(self, df, title_col="Tokenized Title", desc_col="Tokenized Description", predicted_col="Predicted"):
    #     v_title = len(self.vocabulary_title)
    #     v_desc = len(self.vocabulary_desc)
    #     k = len(self.classes)
        
    #     def predict_row(row):
    #         title_tokens = row[title_col]
    #         desc_tokens = row[desc_col]
    #         class_scores = {c: self.phi_y[c] for c in self.classes}
            
    #         for c in self.classes:
    #             for token in title_tokens:
    #                 if token in self.vocabulary_title:
    #                     class_scores[c] += self.phi_x_y_title[c].get(token, 0)
    #                 else:
    #                     class_scores[c] += np.log(self.smoothening / (self.total_tokens_y_title[c] + self.smoothening * v_title))
                
    #             for token in desc_tokens:
    #                 if token in self.vocabulary_desc:
    #                     class_scores[c] += self.phi_x_y_desc[c].get(token, 0)
    #                 else:
    #                     class_scores[c] += np.log(self.smoothening / (self.total_tokens_y_desc[c] + self.smoothening * v_desc))
            
    #         return max(class_scores, key=class_scores.get)
        
    #     df[predicted_col] = df.apply(predict_row, axis=1)

    def fit_dual(self, df, smoothening, class_col="Class Index", title_col="Tokenized Title", desc_col="Tokenized Description"):
        self.classes = df[class_col].unique()
        class_counts = df[class_col].value_counts()
        m = len(df)
        
        for c in self.classes:
            self.phi_y[c] = np.log(class_counts[c] / m)
        
        num_tokens_title = {c: {} for c in self.classes}
        num_tokens_desc = {c: {} for c in self.classes}
        total_tokens_title = {c: 0 for c in self.classes}
        total_tokens_desc = {c: 0 for c in self.classes}
        
        # Initialize a single vocabulary
        self.vocabulary = set()
        
        for _, row in df.iterrows():
            c = row[class_col]
            title_tokens = row[title_col]
            desc_tokens = row[desc_col]
            
            for token in title_tokens:
                self.vocabulary.add(token)
                num_tokens_title[c][token] = num_tokens_title[c].get(token, 0) + 1
                total_tokens_title[c] += 1
            
            for token in desc_tokens:
                self.vocabulary.add(token)
                num_tokens_desc[c][token] = num_tokens_desc[c].get(token, 0) + 1
                total_tokens_desc[c] += 1

        self.total_tokens_y_title = total_tokens_title
        self.total_tokens_y_desc = total_tokens_desc
        self.smoothening = smoothening
        
        v = len(self.vocabulary)  # Combined vocabulary size
        
        for c in self.classes:
            # Use combined vocabulary for both title and description features
            self.phi_x_y_title[c] = {
                token: np.log((num_tokens_title[c].get(token, 0) + smoothening) / 
                (total_tokens_title[c] + smoothening * v)) 
                for token in self.vocabulary
            }
            self.phi_x_y_desc[c] = {
                token: np.log((num_tokens_desc[c].get(token, 0) + smoothening) / 
                (total_tokens_desc[c] + smoothening * v)) 
                for token in self.vocabulary
        }


    def predict_dual(self, df, title_col="Tokenized Title", desc_col="Tokenized Description", predicted_col="Predicted"):
        v = len(self.vocabulary)  # Combined vocabulary size
        
        def predict_row(row):
            title_tokens = row[title_col]
            desc_tokens = row[desc_col]
            class_scores = {c: self.phi_y[c] for c in self.classes}
            
            for c in self.classes:
                for token in title_tokens:
                    if token in self.vocabulary:
                        class_scores[c] += self.phi_x_y_title[c].get(token, 0)
                    else:
                        # Apply smoothing with combined vocabulary size
                        class_scores[c] += np.log(self.smoothening / (self.total_tokens_y_title[c] + self.smoothening * v))
                
                for token in desc_tokens:
                    if token in self.vocabulary:
                        class_scores[c] += self.phi_x_y_desc[c].get(token, 0)
                    else:
                        # Apply smoothing with combined vocabulary size
                        class_scores[c] += np.log(self.smoothening / (self.total_tokens_y_desc[c] + self.smoothening * v))
            
            return max(class_scores, key=class_scores.get)
        
        df[predicted_col] = df.apply(predict_row, axis=1)