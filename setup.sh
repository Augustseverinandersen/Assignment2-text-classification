#!/usr/bin/env bash

# Create virtual environment

# Activate virtual environment

# Installing requirements 
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# Run the code 
python3 src/logestic_reg_classifier.py --filepath ./data/fake_or_real_news.csv
python3 src/neural_network.py --filepath ./data/fake_or_real_news.csv

# Deactivate the venv

