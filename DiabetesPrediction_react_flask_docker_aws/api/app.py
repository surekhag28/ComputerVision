#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 17:15:36 2021

@author: surekhagaikwad
"""

from flask import Flask,request,jsonify,make_response
import pickle
import logging
from flask_cors import CORS
import numpy as np
from flasgger import Swagger


logging.basicConfig(level=logging.INFO)
logger = logging.Logger('Hello World')


filename = 'diabetes-predictor.pkl'

pickle_in = open(filename,'rb')
classifier = pickle.load(pickle_in)

app = Flask(__name__)
#Swagger(app)

cors = CORS(app, resources={r"/predict": {"origins": "http://ec2-3-129-211-233.us-east-2.compute.amazonaws.com:3000"}})


@app.route('/predict',methods=['POST'])
def predict_diabetes():
    
    logger.info('Welcome!!!')
    preg = int(request.form['pregnancies'])
    logger.info(preg)
    glucose = int(request.form['glucose'])
    bp = int(request.form['bloodpressure'])
    st = int(request.form['skinthickness'])
    insulin = int(request.form['insulin'])
    bmi = float(request.form['bmi'])
    dpf = float(request.form['dpf'])
    age = int(request.form['age'])
    
    d = np.array([[preg,glucose,bp,st,insulin,bmi,dpf,age]])
    prediction = classifier.predict(d)
    
    data = {'class':str(prediction[0])}
    response = make_response(jsonify(data))
    
    response.headers['Access-Control-Allow-Origin']='http://ec2-3-129-211-233.us-east-2.compute.amazonaws.com:3000'

    return response


#if __name__=='__main__':
 #   app.run(host='0.0.0.0',port=5000)
     
     
     
     
            
            

