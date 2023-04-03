#!/usr/bin/env bash

# Create virtual enviroment 
python3 -m venv assignment_2

# Activate virtual enviroment 
source ./assignment_2/bin/activate 

# Installing requirements 
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# Run the code 
python3 src/logistic_reg_classifier.py --filepath ./data/fake_or_real_news.csv
python3 src/neural_network.py --filepath ./data/fake_or_real_news.csv

# Deactivate the venv
deactivate