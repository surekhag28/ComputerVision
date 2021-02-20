#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 13:46:29 2021

@author: surekhagaikwad
"""

import io
import base64
from flask import Flask,request,jsonify,make_response
import logging
from PIL import Image
from flasgger import Swagger
from regressor import train

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
            m: coefficient
            c: coefficient
        
    """
    
    logger.info('Welcome to linear regrssion using gradient descent')
    file = request.files.get('file')
    filename = file.filename
    destination = '/'.join([UPLOAD_FOLDER,filename])
    file.save(destination)
    rmse,r2,m,c=train(destination)
    
    grad_img_path = 'uploads/grad.png'
    cost_img_path = 'uploads/cost.png'
    grad_img = get_encoded_img(grad_img_path)
    cost_img = get_encoded_img(cost_img_path)
    
    data = {'root_mean_square_error':rmse,'r_squared':r2,
            'm':m,'c':c,"grad_img": grad_img,"cost_img":cost_img}
    
    response = make_response(jsonify(data))
    
    response.headers['Access-Control-Allow-Origin']='http://ec2-3-22-101-30.us-east-2.compute.amazonaws.com:3000'
    
    return response

def get_encoded_img(image_path):
    img = Image.open(image_path, mode='r')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='png')
    my_encoded_img = base64.encodebytes(img_byte_arr.getvalue()).decode('ascii')
    return my_encoded_img


if __name__=='__main__':
    app.run(host='0.0.0.0',port=5000)
