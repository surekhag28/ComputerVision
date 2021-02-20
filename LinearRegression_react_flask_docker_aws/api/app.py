#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 23:09:27 2021

@author: surekhagaikwad
"""

from flask import Flask,request,jsonify,make_response
import logging
from flasgger import Swagger
from regressor import train,plot

logging.basicConfig(level=logging.INFO)
logger = logging.Logger('Hello World')

UPLOAD_FOLDER = './uploads'

app = Flask(__name__)
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER
Swagger(app)

@app.route('/upload',methods=['POST'])
def fileUpload():
    
    """Let's Predict the category of given image
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            rmse: rmse
            r_squared:  r_squared
        
    """
    
    logger.info('Welcome to train model')
    file=request.files.get('file')
    filename = file.filename
    destination='/'.join([UPLOAD_FOLDER,filename])
    file.save(destination)
    rmse,r_squared,b0,b1=train(destination)
    
    data = {'root_mean_sqaure_error':rmse,'r_squared':r_squared,'b0':b0,'b1':b1}
    response = make_response(jsonify(data))
    
    response.headers['Access-Control-Allow-Origin']='http://ec2-3-22-101-30.us-east-2.compute.amazonaws.com:3000'
    
    return response

@app.route('/get_plot')
def get_plot():
    path = plot()
    data = {'img_path':path}
    response = make_response(jsonify(data))
    
    response.headers['Access-Control-Allow-Origin']='http://ec2-3-22-101-30.us-east-2.compute.amazonaws.com:3000'
    
    return response
    

#if __name__=='__main__':
 #   app.run(host='0.0.0.0',port=5000)
