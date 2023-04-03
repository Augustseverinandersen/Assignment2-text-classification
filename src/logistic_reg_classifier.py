# Importing packages 
# System tools
import os
import sys
sys.path.append(".")
import argparse

# Data munging tools
import pandas as pd

# Importing from Sci-Kit Learn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn import metrics
from joblib import dump, load

# Visualisation


# Defining a function for the user to input a filepath
def input_parse():
    # initialize the parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str) # argument is filepath as a string
    args = parser.parse_args()

    return args

# Loading data
def loading_data(args): 
    print("Loading data")
    data = pd.read_csv(args.filepath, index_col=0) # loading the data as a pandas dataframe 
    return data




# Assigning data
def assigning_data(data): 
    print("assigning data")
    X = data["text"] # column text to variable X
    y = data["label"] # column label to variable Y
    return X, y

    


# Creating train test split of 80/20
def split(text, label):
    print("creating 80/20 split")
    X_train, X_test, y_train, y_test = train_test_split(text,          # texts for the model
                                                        label,          # classification labels
                                                        test_size=0.2,   # create an 80/20 split
                                                        random_state=42) # chosing not to get a random split everytime
    return X_train, X_test, y_train, y_test





# Vectorizing and Feature Extraction 
def vectorizer_function():
    print("Vectorizing and feature extraction")
    vectorizer = CountVectorizer(ngram_range = (1,2),     # unigrams and bigrams (1 word and 2 word units)
                                lowercase =  True,       # why use lowercase?
                                max_df = 0.95,           # remove very common words
                                min_df = 0.05,           # remove very rare words
                                max_features = 100)      # keep only top 100 features
    return vectorizer




# fit_transforming training data 
def transform(text_train, text_test, vectorizer):
    print("Fitting training data")
    X_train_feats = vectorizer.fit_transform(text_train) # fit_transform = taking all input text and fitting to above parameters. 
    #transforming test data 
    print("transforming test data")
    X_test_feats = vectorizer.transform(text_test) # transform without fit here to see if it works in the real world. 

    return X_train_feats, X_test_feats


# get feature names
def feature(vectorizer):
    feature_names = vectorizer.get_feature_names_out() # Storing all the feature names, which can be unigrams and bigrams 
    return feature_names

# Classifying 
def classifier_function(text_train_feats, label_train, text_test_feats):
    print("Classifying")
    classifier = LogisticRegression(random_state=42).fit(text_train_feats, label_train) # Using LogisticReg to match label with text
    y_pred = classifier.predict(text_test_feats) # and storing in y_pred
    return classifier, y_pred



# Checking the performance of the model with f1
def performance(label_test, label_prediction):
    print("Making performance model:")
    classifier_metrics = metrics.classification_report(label_test, label_prediction) # Checking to see how the true labels compare with our predictions
    print(classifier_metrics)
    return classifier_metrics

# Saving report
def report_save_function(classifier_metrics):
    folder_path = os.path.join(".", "out")
    file_name = "logestic_reg_classifier_metrics.txt"
    file_path = os.path.join(folder_path, file_name)

    with open(file_path, "w") as f: # "Writing" the classifier metrics, thereby saving it.
        f.write(classifier_metrics)
    print("Reports saved")

# Saving models
def model_save_function(classifier, vectorizer):
    dump(classifier, "models/logestic_reg_LR_classifier.joblib") # Saving the models in folder models as a joblib file.
    dump(vectorizer, "models/logestic_reg_tfidf_vectorizer.joblib")
    print("Models saved")


def main_function():
    print("Logistic Regression Classifier Script:")
    args = input_parse()
    data = loading_data(args)
    X, y = assigning_data(data)
    X_train, X_test, y_train, y_test = split(X, y)
    vectorizer = vectorizer_function()
    X_train_feats, X_test_feats = transform(X_train, X_test, vectorizer)
    feature_names = feature(vectorizer)
    classifier, y_pred = classifier_function(X_train_feats, y_train, X_test_feats)
    classifier_metrics = performance(y_test, y_pred)
    report_save_function(classifier_metrics)
    model_save_function(classifier, vectorizer)

if __name__ == "__main__":
    main_function()