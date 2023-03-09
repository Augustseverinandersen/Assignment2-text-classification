# Importing packages 
# System tools
import os
import sys
sys.path.append(".")
import argparse

# Data munging tools
import pandas as pd
import utils.classifier_utils as clf

# Importing from Sci-Kit Learn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn import metrics

# Visualisation
import matplotlib.pyplot as plt

# Defining a function for the user to input a filepath
def input_parse():
    # initialize the parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str) # argument is filepath as a string
    args = parser.parse_args()

    filepath = args.filepath
    return filepath
filepath = input_parse() # the functions output to be stored in filepath

# Loading data 
print("Loading data")
data = pd.read_csv(filepath, index_col=0) # loading the data as a pandas dataframe 

# Assigning data 
print("assigning data")
X = data["text"] # column text to variable X
y = data["label"] # column label to variable Y

# Creating train test split of 80/20
print("creating 80/20 split")
X_train, X_test, y_train, y_test = train_test_split(X,          # texts for the model
                                                    y,          # classification labels
                                                    test_size=0.2,   # create an 80/20 split
                                                    random_state=42) # chosing not to get a random split everytime

# Vectorizing and Feature Extraction 
print("Vectorizing and feature extraction")
vectorizer = CountVectorizer(ngram_range = (1,2),     # unigrams and bigrams (1 word and 2 word units)
                             lowercase =  True,       # why use lowercase?
                             max_df = 0.95,           # remove very common words
                             min_df = 0.05,           # remove very rare words
                             max_features = 100)      # keep only top 100 features

# fit_transforming training data 
print("Fitting training data")
X_train_feats = vectorizer.fit_transform(X_train) # fit_transform = taking all input text and fitting to above parameters. 

#transforming test data 
print("transforming test data")
X_test_feats = vectorizer.transform(X_test) # transform without fit here to see if it works in the real world. 

# get feature names
feature_names = vectorizer.get_feature_names_out() # Storing all the feature names, which can be unigrams and bigrams 

# Classifying 
print("Classifying")
classifier = LogisticRegression(random_state=42).fit(X_train_feats, y_train) # Using LogisticReg to match label with text
y_pred = classifier.predict(X_test_feats) # and storing in y_pred

# Checking the performance of the model with f1
print("Making performance model")
classifier_metrics = metrics.classification_report(y_test, y_pred) # Checking to see how the true labels compare with our predictions

# Saving report
folder_path = os.path.join(".", "out")
file_name = "logestic_reg_classifier_metrics.txt"
file_path = os.path.join(folder_path, file_name)

with open(file_path, "w") as f: # "Writing" the classifier metrics, thereby saving it.
    f.write(classifier_metrics)
print("Reports saved")

# Saving models
from joblib import dump, load
dump(classifier, "models/logestic_reg_LR_classifier.joblib") # Saving the models in folder models as a joblib file.
dump(vectorizer, "models/logestic_reg_tfidf_vectorizer.joblib")
print("Models saved")

