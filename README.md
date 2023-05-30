# 2. Assignment 2 - Text classification benchmarks
## 2.1 Assignment Description
Written by Ross:

For this exercise, you should write two different scripts. One script should train a logistic regression classifier on the data; the second script should train a neural network on the same dataset. Both scripts should do the following:
- Be executed from the command line.
- Save the classification report to the folder called out.
- Save the trained models and vectorizers to the folder called models.
## 2.2 Machine Specifications and My Usage
All the computation done for this project was performed on the UCloud interactive HPC system, which is managed by the eScience Center at the University of Southern Denmark. The scripts were created with Coder Python 1.73.1 and Python version 3.9.2. 
### 2.2.1 Perquisites
To run the scripts, make sure to have Bash and Python3 installed on your device. The script has only been tested on Ucloud.
## 2.3 Contribution
This assignment was made in contribution with fellow students from class. All in-code comments are made by me. The dataset used in this assignment is from [Jillani Soft Tech](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news).
### 2.3.1 Data 
This assignment uses a dataset from [Kaggle](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news), which consist of four columns, _number, title, text, or label_. The dataset contains news titles and texts categorised as either fake or real. There are over 6,000 rows of news articles written in English, and it can be deduced that the articles most likely are from an English media. 
## 2.4 Packages
This assignment uses the following packages:
- **Os** is used to navigate across operating systems.
- **Sys** is used to navigate the directory.
- **Argparse** is used to create command-line arguments. In this assignment, it is used to define the path to the zip file.
- **Zipfile** is used to extract the zip file.
- **Pandas (version 1.5.3)** is used to read the data.
- **Scikit-Learn (version 1.2.2)** is used to import the following: _CountVectorizer_ creates tokens, which are used as features. _LogisticRegression_ is an algorithm from Scikit-learn and is used to classify the texts. _MLPClassifier_ is used to create the neural network. _Train_test_split_ is used to split the data into training data and testing data. _Metrics_ is used to create a classification report.
- **Joblib (version 1.2.0)** is used to save the models and the vectorizer.
## 2.5 Repository Contents
This repository contains the following folders and files:
- ***data*** is an empty folder where the zip file will be stored.
- ***models*** is the folder where the saved models and vectorizers for both scripts will be stored.
- ***out*** is the folder where the classification reports for both scripts will be stored.
- ***src*** is the folder that contains the two scripts ``logistic_reg_classifier.py`` and ``neural_network.py``.
- ***README.md*** is the readme file.
- ***requirements.txt*** is a text file with version-controlled packages that need to be installed.
- ***setup.sh*** is the file that creates a virtual environment, upgrades pip, and installs the packages from requirements.txt.
## 2.6 Methods
**Script logistic_reg_classifier.py:**
- The script starts by initializing an argparse, which is used to define the path to the zip file and to set the number of _max_features_ in the function ``vectorizer_function``.
- The zip file is then unpacked, and the data is loaded into a data frame using Pandas.
- The column's text and label are then stored in the variables X and y. 
- The next function, ``split``, uses Scikit-Learns _train_test_split_ to store 20% of the data as test data, and the remaining as train data.
- The function ``vectorizer_function`` uses _CountVectorizer_ from Scikit-Learn to tokenize the words with the following criteria specified: All words are set to lowercase, words can be unigrams and bigrams, the 5% most common and rare words are removed, and only the top 100 features are kept. 
- The training data is then fitted to the ``vectorizer_function`` and stored as a feature vector. The test data is not fitted but is only stored as a feature vector, to resemble the real world.
- The logistic regression is then trained on the training data to match labels with the text. And the trained classifier is then tested on the test data. 
- A classification report is then created on the true labels and the predictions to see how the classifier performed. 
- Lastly, the classification report is saved to the folder _out_, and the model and the vectorizer are saved to folder _models_.

**Script neural_network.py:**
- This script starts by loading, the already unzipped, data in as a Pandas data frame.
- The column's _text_ and _label_ are then assigned to variables X and y.
- A _train_test_split_ is created of 80% training data and 20% test data. 
- A vectorizer is created with the same arguments as in the logistic regression script, except that this time the top 500 features are kept. 
- The training data is then fitted to the vectorizer and stored as feature vectors, and the test data is just stored as feature vectors. 
- The next step in the script is the creation of the neural network. This is done in the function ``classifier_architecture``, using Scikit-Learns _MLPClassifier_. The neural network uses the _activation_ function _logistic_, which maps the input between 0 and 1. It has one hidden layer with 20 neurons. The max iteration is set at 1000, which means the neural network stops after 1000 iterations. Lastly, the neural network has a set seed.
- The model is then trained on the training data and tested on the test data. 
- A classification report of the predictions for the test data and the true labels are then made, to see the performance of the model.
- Lastly, the classification report is saved to the folder _out_, and the model and vectorizer are saved to the folder _models_.
## 2.7 Discussion 
The logistic regressions classification report shows an _accuracy f1-score_ of 0.82, with the _fake(f1-score: 0.83)_ performing slightly better than _real(f1-score 0.81)_. The neural network classification report shows an _accuracy f1-score of 0.89_, with the _real(f1-score 0.89)_ marginally performing better than the _fake(f1-score 0.88)_, 

Both the logistic regression and the neural network have a high _accuracy f1-score_. The logistic regression works well since there are only two classes to predict. The logistic regression can thereby learn which words and patterns are used in the fake or real data, and create predictions based on that. The neural network works a bit better than the logistic regression, as it can be more fine-tuned to the data by updating its weights. Furthermore, the logistic regression script has the _CountVectorizer_ argument _max_features_ set to 100 and the neural network script has it set to 500. When I change the _max_features_ for the logistic regression this was the performance change:

| **Max_features** | **Accuracy f1-score** |
|--------------|------------------:|
| 100          |              0.82 |
| 200          |              0.85 |
| 300          |              0.87 |
| 500          |              0.88 |
| 800          |              0.89 |
| 1000         |              0.90 |
| 2000         |              0.91 |

This is interesting, as the logistic regression gets more features it performs better. My original hypothesis would have been that too many features would have caused the fake and real news to become too similar to separate. 

For the neural network having more features to learn about is a strength, since it can update its weights accordingly to give the best predictions. However, too many features could become a problem as the model would start to overfit. 
## 2.8 Usage
To run this code, follow these steps:

**OBS! Run ``logistic_reg_classifier.py`` first as this is the script that unzips the zip file.**
- Clone the repository.
- Navigate to the correct directory.
- Get the zip file, from [here](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news), and place it in the _data_ folder (you might need to rename the zip file).
- Run ``bash setup.sh`` in the command line. This will install the requirements and create a virtual environment.
- Run ``source ./assignment_2/bin/activate`` in the command line to activate the virtual environment.
- Run ``python3 src/logistic_reg_classifier.py --zip_path data/zip_name.zip --features 100`` in the command line to run the ``logistic_reg_classifier.py`` script.
    - The argparse ``--zip_path`` takes a string as input and is used to define the path to the zip file.
    - The argparse ``--features`` takes an integer as input, and has a default of 100. Only include this argparse if you want experiment with more or less _max_features_.
- Run ``python3 src/neural_network.py`` in the command line to run the ``neural_network.py`` script.
- The models will be saved in the folder models and the classification reports will be saved in the folder out.

