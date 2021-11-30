# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 18:04:06 2021

@author: anil.ms
"""
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


#create flask app
app = Flask(__name__)

#load model pickle
model = pickle.load(open('model.pickle', 'rb'))

#home page - by default route is slash - renders the template.html
@app.route('/')
def home():
    return render_template('index.html')

#provide features to model.pickle in order to make predictions - '/predict' will hit the predict function
@app.route('/predict', methods=['POST'])
def predict():
    input_feats = [x for x in request.form.values()]
    #final_feats = [np.array(input_feats)]
    prediction = model.predict(np.array(input_feats).reshape(1,-1))
    
    #output=round(prediction[0],2)
    
    return render_template('index.html', predictions='categorized as class {}'.format(str(prediction)))

#main function will run the whole flask
if __name__ == '__main__':
    app.run(debug=True)
    

