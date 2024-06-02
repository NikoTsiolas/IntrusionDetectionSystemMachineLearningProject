# Intrusion Detection System Machine Learning Project
A simple machine learning/ Cyber security project that is an intrusion detection system based on NSL KDD test and train data. 


## Project Overview
This project is focused on developing a machine learning model to detect network intrusions. The model uses the NSL-KDD dataset for training and testing, applying a RandomForestClassifier to distinguish between normal and malicious activities in network traffic.

## Features
- Use of RandomForestClassifier for effective anomaly detection.
- Data preprocessing includes one-hot encoding of categorical features and scaling of numerical features.
- Evaluation of model performance using accuracy, precision, recall, and F1-score.

## Getting Started

### Prerequisites
- Python 3.8+
- Pandas
- NumPy
- scikit-learn
- Flask
- Joblib

### Installation
Clone the repository to your local machine:
```bash
git clone https://github.com/NikoTsiolas/IntrusionDetectionSystemMachineLearningProject.git
cd IntrusionDetectionSystemMachineLearningProject
Install the required packages:

bash
Copy code
pip install -r requirements.txt
Usage
To run the Flask API:

bash
Copy code
python app.py
This will start the server on localhost:10001 and you can send POST requests to /predict to get predictions.

Model Training
To train the model, run:

bash

python model_training.py
This script will preprocess the data, train the model, and save it as model.pkl for the API to use.

Contributing
Contributions to this project are welcome. Please fork the repository and submit a pull request with your features or fixes.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Contact
Niko Tsiolas - @Thestuze on instagram, email- nikotsiolas@gmail.com, twitter: @thestuze

Acknowledgments
Thanks to the NSL-KDD dataset providers.
Appreciation for the open-source libraries used in this project.


