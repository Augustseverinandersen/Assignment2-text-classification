# Importing librarys 

# system tools
import os
import sys
sys.path.append(".")

# data munging tools
import pandas as pd

# Importing from Sci-kit learn
from sklearn.feature_extraction.text import CountVectorizer #, TfidfVectorizer Removeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn import metrics

# Saving the model and vectoriser
from joblib import dump, load


# Reading the data 
def loading_data(): 
    print("Loading data")
    folder_path = os.path.join("data", "fake_or_real_news.csv") # Specifying folder path
    data = pd.read_csv(folder_path, index_col=0) # loading the data as a pandas dataframe 
    return data


# Assigning the data
def assigning_data(data): 
    print("Assigning data")
    X = data["text"] # column text to variable X
    y = data["label"] # column label to variable Y
    return X, y


# Creating train test split of 80/20
def split(text, label):
    print("Creating 80/20 split")
    X_train, X_test, y_train, y_test = train_test_split(text,          # texts for the model
                                                        label,          # classification labels
                                                        test_size=0.2,   # create an 80/20 split for train test
                                                        random_state=42) # chosing not to get a random split everytime
    return X_train, X_test, y_train, y_test


# Vectorizing and Feature Extraction 
def vectorizer_function():
    print("Vectorizing and feature extraction")
    vectorizer = CountVectorizer(ngram_range = (1,2),     # unigrams and bigrams (1 word and 2 word units)
                                lowercase =  True,       # Setting all to lowercase
                                max_df = 0.95,           # remove very common words
                                min_df = 0.05,           # remove very rare words
                                max_features = 500)      # keep only top 500 features
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


# Classifying and predicting 
def classifier_architecture():
    print("Classifying and predicting")
    classifier = MLPClassifier(activation = "logistic", # logistic = giving a value between 0-1 to the data
                            hidden_layer_sizes = (20,), # one hidden layer with 20 neurons
                            max_iter=1000, # setting a limit of 1000 iterations
                            random_state = 42) # making it reproducable
    return classifier


def train_model(classifier, text_train_feats, label_train):
    print("Training model")
    model = classifier.fit(text_train_feats, label_train) # training the model to get the most accurate prediction
    return model

# Getting predictions
def prediction(model, text_test_feats):
    print("Creating predictions")
    y_pred = model.predict(text_test_feats) # Generating predictions for the test data
    return y_pred

# Calculating metrics 
def classifier_report(label_test, y_pred):
    print("Classifier Metrics:") # Checking to see how the true labels compared with our predictions
    classifier_metrics = metrics.classification_report(label_test, y_pred) 
    print(classifier_metrics)
    return classifier_metrics

def metrics_save_function(classifier_metrics):
    folder_path = os.path.join("out")
    file_name = "neural_networks_classifier_metrics.txt"
    file_path = os.path.join(folder_path, file_name) # Defining where to save the classifer report.

    with open(file_path, "w") as f: # "Writing" the classifier metrics, thereby saving it.
        f.write(classifier_metrics)
    print("Reports saved")


# Saving models
def model_save_function(classifier, vectorizer):
    dump(classifier, "models/neural_network_LR_classifier.joblib") # Saving the models in folder models as a joblib file.
    dump(vectorizer, "models/neural_network_tfidf_vectorizer.joblib")


def main_function():
    data = loading_data() # Reading data as pandas dataframe
    X, y = assigning_data(data) # Assigning data
    X_train, X_test, y_train, y_test = split(X, y) # Splitting data
    vectorizer = vectorizer_function() # Parameters for vectorizing
    X_train_feats, X_test_feats = transform(X_train, X_test, vectorizer) # Transforming the data 
    feature_names = feature(vectorizer) # Getting features
    classifier = classifier_architecture() # The model architecture
    model = train_model(classifier, X_train_feats, y_train) # Training the model
    y_pred = prediction(model, X_test_feats) # Getting predictions
    classifier_metrics = classifier_report(y_test, y_pred) # Classification report
    metrics_save_function(classifier_metrics) # Saving report
    model_save_function(classifier, vectorizer) # Saving model


if __name__ == "__main__": # If this script is called from the terminal run main function
    main_function()
