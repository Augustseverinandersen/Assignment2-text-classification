[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10362462&assignment_repo_type=AssignmentRepo)
# Assignment 2 - Text classification benchmarks

Github link: ```Github link```

## Contribution
- This assignment was made in contribution with fellow students from class, and with code from the in-class notebooks. All in-code comments are made by me.
- This assignment uses data from [Kaggle](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news), which consits of four columns with either a fake or real news article. The four columns are *Number, Title, Text,* and *Label*. 

## Packages 
- This assignment uses the following packages:
    - Os
        - Is used to navigate the operating system.
    - Sys
        - Is used to navigate the directory. 
    - Argparse
        - Is used to create command-line arguments.
    - Zipfile
        - Is used to extract zip files.
    - Pandas
        - Pandas is used to structure the data and manipulation.
    - Sklearn
        - Scikit learn is used to get packages created for making models, feature extraction, logistic regression, classifying, and creating a train test split.
    - Joblib
        - Joblib is used to save the models.
## Assignment description 
Written by Ross:
For this exercise, you should write *two different scripts*. One script should train a logistic regression classifier on the data; the second script should train a neural network on the same dataset. Both scripts should do the following:

- Be executed from the command line
- Save the classification report to the folder called ```out```
- Save the trained models and vectorizers to the folder called ```models```

## Methods / What does the code do
- Script *logistic_reg_classifier.py*:
    - The code unpacks the zip file, and loads it as a pandas dataframe. Assigns the columns text and label to X and y. Creates a train test split of 80/20. Creates parameters for the training data, so the most and least common words are removed, and only the top 100 features are kept. Transforms the data with the created parameters. Classifies the data, before testing the performance on the test data. Lastly, the model and classification reports are saved.
- Script *neural_network.py*:
    - Unzips the data, places it in a dataframe, assigns the data, and creates a split. The parameters created here are the same expects that this scripts keeps the top 500 features. The script than transforms the data and creates the model architecture. The architecture here uses a logistic activation, with one hidden layer of 20 neurons, has a max iteration of 1000, and has a set random state. The model is than trained before testing it on the test data. A classification report is created and saved, and the model is saved aswell.
## Discussion 
- When I ran the script my logistic regression script had an accuracy f1-score of 0.82, while the neural_network script had a score of 0.89 The neural network performed slightly better on the REAL data, but the logistic regression script performed better on the FAKE data.
- Both the logistic regression and the neural network have a very high f1 accuracy score. The logistic regression works well since there are only two classes it has to predict. The logistic regression can thereby learn which words and patterns are used in the FAKE or REAL data, and create predictions based on that. The neural network works a bit better than the logistic regression, but is harder to decode what it is learning on. However, it seems, that the neural network can find more precise patterns than the logistic regression.

## Usage 
To run this code foloow these steps:
- Clone the repository
- Get the zip file, from [here](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news), and place it in the data folder. 
- Run ```bash setup.sh``` in the command line. This will install the requirements, and create a virtual environment. 
- Run ```source ./assignment_2/bin/activate ``` in the command line to activate the virtual environment. 
- Run ```python3 scr/logistic_reg_classifier.py --zip_path data/zip_name.zip``` in the command line to run the logistic_reg_classifier script.
- Run ```python3 scr/neural_network.py --zip_path data/zip_name.zip``` in the command line to run the neural_network script.

__OBS! Replace *zip_name.zip* with your zip file__ 
- The models will be saved in the folder *models* and the classification reports will be saved in the folder *out*
