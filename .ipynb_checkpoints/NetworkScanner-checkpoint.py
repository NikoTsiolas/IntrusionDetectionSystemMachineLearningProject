#NetworkScanner.py
#Author: Niko Tsiolas
#Date: 05/15/2024


import pandas as pd 

#load the training data
train_data = pd.read_csv('NSL_KDD_Train.csv')

#load the testing data
testing_data = pd.read_csv('NSL_KDD_Test.csv')

train_data.head()

train_data.describe()

train_data.info()