import os
import json
from flasgger import Swagger
from flask import Flask, render_template, request, redirect, url_for, abort, \
    send_from_directory,jsonify
from PIL import Image
    

from torch_utils import transform_image, get_prediction

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.jpeg']
app.config['UPLOAD_PATH'] = './uploads'

Swagger(app)


class_index_name = json.load(open('class_index_name.json'))

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    # xxx.png
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    file = 'img1.jpeg'
    return render_template('index.html', file=file)

@app.route('/', methods=['POST'])
def upload_files():
    uploaded_file = request.files['file']
    filename = uploaded_file.filename
    if filename != '':
        uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
    
    #img_bytes = uploaded_file.read()
    img = Image.open('./uploads/img1.jpeg')
    tensor = transform_image(img)
    prediction = get_prediction(tensor)
    predicted_idx = str(prediction.item())
    print(predicted_idx)
            
    data = {'prediction': predicted_idx, 'class_name': class_index_name[predicted_idx]}
    return render_template('index.html', file=uploaded_file.filename, category=class_index_name[predicted_idx])

@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename)


if __name__=='__main__':
    app.run(host='0.0.0.0',port=8000)