[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10362462&assignment_repo_type=AssignmentRepo)
# Assignment 2 - Text classification benchmarks

Github link: ```Github link```

## Requirements 
For this exercise, you should write *two different scripts*. One script should train a logistic regression classifier on the data; the second script should train a neural network on the same dataset. Both scripts should do the following:

- Be executed from the command line
- Save the classification report to the folder called ```out```
- Save the trained models and vectorizers to the folder called ```models```

## How to run this code 

- Clone the repository
- Open terminal in directory 
- write ```bash setup.sh```

## What does this directory contain 

This directory contains:
- Two python scripts 
    - *logestic_reg_classifier.py*: The script for training a logestic regression classifier.
    - *neural_network.py*: The script for training a neural netwok.
- One shell script 
    - The shell script first upgrades pip
    - Than installs requirements.txt
    - Then runs the two scripts, with the filepath specified.
- One requirements text file
    - Contains all librarys to be installed.
- Folder __models__ for saved models 
- Folder __out__ for saved classification reports
- Folder __data__ which contains a csv file 
    - Contains four columns 
    1. index
    2. title 
    3. text 
    4. label
- Folder __utils__ which contains Ross's work

## What does this code do 

The code in this directory trains four models. Two are trained with a logestic regression classifier, and two are trained with a neural network. The data that is being tested is a csv file containing real and fake news. For a deeper understanding of the code line for line, look at the comments in the scripts.

## Which librarys are being installed 
- Pandas 
- Sci-kit learn 
- Numpy 
- Joblib 
- Argparse 
- Seaborn 
- Matplotlib 
```add what the packages do```
