
"""
Created on Fri Jul 10 21:47:42 2020

@author: sharm
"""


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
     
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    output = str(prediction)
    
    
    return render_template('index.html', prediction_text='THIS CASE IS DIAGONISED AS {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = str(prediction)
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)