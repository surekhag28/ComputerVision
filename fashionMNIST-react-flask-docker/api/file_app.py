import os
import json
from flask import Flask, flash, request, redirect, url_for, session, jsonify, make_response
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
from PIL import Image
import logging

from torch_utils import transform_image,get_prediction

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger('HELLO WORLD')

class_index_name = json.load(open('class_index_name.json'))

UPLOAD_FOLDER = './uploads/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)

cors = CORS(app, resources={r"/upload": {"origins": "http://ec2-18-217-62-181.us-east-2.compute.amazonaws.com:3000/"}})

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#app.config['CORS_HEADERS'] = 'Content-Type'



@app.route('/upload', methods=['POST'])
#@cross_origin()
#@crossdomain(origin='http://ec2-18-217-94-214.us-east-2.compute.amazonaws.com:3000',headers=['Access-Control-Allow-Origin','Content-Type'])
def fileUpload():
    target=os.path.join(UPLOAD_FOLDER,'test_docs')
    if not os.path.isdir(target):
        os.mkdir(target)
    logger.info("welcome to upload`")
    file = request.files['file'] 
    filename = secure_filename(file.filename)
    destination="/".join([target, filename])
    file.save(destination)
    #session['uploadFilePath']=destination

    img = Image.open('./uploads/test_docs/img1.jpeg')
    tensor = transform_image(img)
    prediction = get_prediction(tensor)
    predicted_idx = str(prediction.item())
    print(predicted_idx)
            
    data = {'prediction': predicted_idx, 'class_name': class_index_name[predicted_idx]}
    response = make_response(jsonify(data));

    # Add Access-Control-Allow-Origin header to allow cross-site request
    response.headers['Access-Control-Allow-Origin'] = 'http://ec2-18-217-62-181.us-east-2.compute.amazonaws.com:3000';


    return response

