# Importing librarys 

# system tools
import os
import sys
sys.path.append(".")
import argparse

# data munging tools
import pandas as pd
import utils.classifier_utils as clf

# Importing from Sci-kit learn
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
filepath = input_parse() # the Argparse function to be stored in filepath 

# Reading the data 
print("Reading data")
data = pd.read_csv(filepath, index_col=0) # Putting the file in a pandas dataframe.

# Assigning the data 
X = data["text"] # Column text to variable X
y = data["label"] # Column label to variable y

# Defining train test split of 80/20
print("Defining test split")
X_train, X_test, y_train, y_test = train_test_split(X,           # texts for the model
                                                    y,          # classification labels
                                                    test_size=0.2,   # create an 80/20 split
                                                    random_state=42) # random state for reproducibility

# Vectorizing and Feature Extracting 
print("Vectorizing")
vectorizer = TfidfVectorizer(ngram_range = (1,2),     # unigrams and bigrams (1 word and 2 word units) 
                             lowercase =  True,       # why use lowercase?
                             max_df = 0.95,           # remove very common words
                             min_df = 0.05,           # remove very rare words
                             max_features = 500)      # keep only top 500 features

# fit_transforming training data 
print("transforming training data")
X_train_feats = vectorizer.fit_transform(X_train) # fit_transform = taking all input text and fitting to above parameters. 

# transforming test data 
print("transforming test data")
X_test_feats = vectorizer.transform(X_test) # transform without fit here to see if it works in the real world. 

# Getting the feature names, after vectorizing the data.
feature_names = vectorizer.get_feature_names_out()

# Classifying and predicting 
print("Classifying and predicting")
classifier = MLPClassifier(activation = "logistic", # logistic = giving a value between 0-1 to the data
                           hidden_layer_sizes = (20,), # number of neurons = 20
                           max_iter=1000, # setting a limit of 1000
                           random_state = 42) # making it reproducible

classifier.fit(X_train_feats, y_train) # training the model to get the most accurate prediction

# Getting predictions
y_pred = classifier.predict(X_test_feats) 

# Calculating metrics 
classifier_metrics = metrics.classification_report(y_test, y_pred) # Checking to see how the true labels compare with our predictions

folder_path = os.path.join(".", "out")
file_name = "neural_networks_classifier_metrics.txt"
file_path = os.path.join(folder_path, file_name)

with open(file_path, "w") as f: # "Writing" the classifier metrics, thereby saving it.
    f.write(classifier_metrics)
print("Reports saved")


# Saving models
from joblib import dump, load
dump(classifier, "models/neural_network_LR_classifier.joblib") # Saving the models in folder models as a joblib file.
dump(vectorizer, "models/neural_network_tfidf_vectorizer.joblib")