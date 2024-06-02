#Niko Tsiolas 
#Saturday Jun 1st 2024
#Flask API for the model


from flask import Flask, request, jsonify
import pandas as pd 
from sklearn import joblib


app = Flask(__name__)


#load the model

model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])

def predict ():
    data = request.get_json()
    df = pd.DataFrame(data, index=[0])
    prediction = model.predict(df)
    return jsonify(prediction.tolist())


if __name__ == '__main__':
    app.run(port=10001, debug=True)