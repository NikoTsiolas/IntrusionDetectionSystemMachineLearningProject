#NetworkScanner.py
#Author: Niko Tsiolas
#Date: 05/15/2024


import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib


#There was an absence of columns so I added them myself
column_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label"
]


#load the training data
train_data = pd.read_csv('NSL_KDD_Train.csv', header=None, names=column_names)

#load the testing data
testing_data = pd.read_csv('NSL_KDD_Test.csv', header=None, names=column_names)

# Histogram of the 'duration' feature

#going to apply the log transformation here to help with interpretability

#needed to do this so that the log transformation would work and never be at 0
adjusted_duration = train_data['duration'] + 0.01

#just taking the log of the adjusted durations
log_duration = np.log10(adjusted_duration)

plt.figure(figsize=(12, 8))  # Sets the figure size
plt.hist(train_data['duration'], bins=50, color='blue', alpha=0.7)  # Creates a histogram
plt.title(' Log-Transformed Histogram of Connection duration')  # Adds a title to the histogram
plt.xlabel('Log-Transformed Duration')  # Label for the x-axis
plt.ylabel('Frequency')  # Label for the y-axis
plt.grid(True)  # Adds a grid for easier readability
plt.show()  # Displays the plot

flags = train_data['flag'].value_counts()  # Counts the occurrence of each flag

plt.figure(figsize=(12, 8))  # Sets the figure size
flags.plot(kind='bar', color='blue', alpha=0.7)  # Creates a bar graph for categorical datas
plt.title('Histogram of Connection flag')  # Adds a title to the histogram
plt.xlabel('Flag Type')  # Label for the x-axis
plt.ylabel('Frequency')  # Label for the y-axis
plt.grid(True)  # Adds a grid for easier readability
plt.show()  # Displays the plot

print(train_data.isnull().sum())
print(testing_data.isnull().sum())



def plot_categorical_data(data,column, title, xlabel,ylabel):
    counts = data[column].value_counts()  # Counts the occurrence of each flag
    plt.figure(figsize=(12, 8))  # Sets the figure size
    counts.plot(kind='bar', color='blue', alpha=0.7)  # Creates a bar graph for categorical datas
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)  # Adds a grid for easier readability
    plt.show()


#select categorical columns

categorical_columns = ['protocol_type', 'service', 'flag']

#One-Hot-Encoding for the categorical columns

one_hot_encoded_data = pd.get_dummies(train_data, columns=categorical_columns)

print(one_hot_encoded_data.head())


#initalize the scaler

scaler = StandardScaler()


#select numerical columns for scaling ( excluding already one hot encoded columns)
numerical_columns = ["duration", "src_bytes", "dst_bytes", "wrong_fragment", "urgent"]

#fit and transform the data 

one_hot_encoded_data[numerical_columns] = scaler.fit_transform(one_hot_encoded_data[numerical_columns])


print(one_hot_encoded_data[numerical_columns].head())


#initalizing the label encoder 

label_encoder = LabelEncoder()


# encode target variable

one_hot_encoded_data['label'] = label_encoder.fit_transform(one_hot_encoded_data['label'])

print(one_hot_encoded_data['label'].head())


#define features and target 

x = one_hot_encoded_data.drop('label', axis=1)
y = one_hot_encoded_data['label']


#split the data 

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


print(f'Training samples: {X_train.shape[0]}')
print(f'Testing samples: {X_test.shape[0]}')

#initialize the random forest classifier

clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

print(classification_report(y_test, y_pred, zero_division=1))






joblib.dump(clf, 'model.pkl')