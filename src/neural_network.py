# Importing librarys 

# system tools
import os
import sys
sys.path.append(".")
import argparse

# data munging tools
import pandas as pd
import utils.classifier_utils as clf

# Machine learning stuff
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn import metrics

# Visualisation
import matplotlib.pyplot as plt

# Arg parse 
def input_parse():
    # initialize the parser
    parser = argparse.ArgumentParser()
    # add arguments // "--filepath" 
    parser.add_argument("--filepath", type=str)
    # parse the arguments from command line
    args = parser.parse_args()

    filepath = args.filepath
    return filepath
filepath = input_parse()

# Reading the data 
print("Reading data")
data = pd.read_csv(filepath, index_col=0)

# Assigning the data 
X = data["text"]
y = data["label"]

# Defining train test split of 80/20
print("Defining test split")
X_train, X_test, y_train, y_test = train_test_split(X,           # texts for the model
                                                    y,          # classification labels
                                                    test_size=0.2,   # create an 80/20 split
                                                    random_state=42) # random state for reproducibility

# Vectorizing and Feature Extracting 
print("Vectorizing")
vectorizer = TfidfVectorizer(ngram_range = (1,2),     # unigrams and bigrams (1 word and 2 word units) // include single tokens New York = 2 tokens and 1 token
                             lowercase =  True,       # why use lowercase?
                             max_df = 0.95,           # remove very common words
                             min_df = 0.05,           # remove very rare words
                             max_features = 500)      # keep only top 500 features

# first we fit the vectorizer to the training data...
print("transforming training data")
X_train_feats = vectorizer.fit_transform(X_train)

#... then transform our test data
print("transforming test data")
X_test_feats = vectorizer.transform(X_test)

# get feature names if needed
feature_names = vectorizer.get_feature_names_out()

# Classifying and predicting 
print("Classifying and predicting")
classifier = MLPClassifier(activation = "logistic",
                           hidden_layer_sizes = (20,),
                           max_iter=1000,
                           random_state = 42)
classifier.fit(X_train_feats, y_train)

# Getting predictions
y_pred = classifier.predict(X_test_feats) 

# Calculating metrics 
classifier_metrics = metrics.classification_report(y_test, y_pred)

folder_path = os.path.join(".", "out")
file_name = "neural_networks_classifier_metrics.txt"
file_path = os.path.join(folder_path, file_name)

with open(file_path, "w") as f:
    f.write(classifier_metrics)
print("Reports saved")


# Saving models
from joblib import dump, load
dump(classifier, "models/neural_network_LR_classifier.joblib")
dump(vectorizer, "models/neural_network_tfidf_vectorizer.joblib")